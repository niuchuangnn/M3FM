import torch
import torch.utils.checkpoint
from torch import nn, Tensor
import math


class TextTransformerConfig:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.layer_norm_eps = 1e-05
        self.max_position_embeddings = 514
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.pad_token_id = 1
        self.type_vocab_size = 1
        self.vocab_size = 50265


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class TextEmbeddings(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids):
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        token_type_ids = buffered_token_type_ids_expanded

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None
    ):

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        outputs = self.feed_forward_chunk(self_attention_outputs)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TextLayer(config) for _ in range(config.num_hidden_layers)])


class TextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TextEmbeddings(config)
        self.encoder = TextEncoder(config)

    def get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        txt_embeds = self.embeddings(input_ids=input_ids)
        attn_txt_mask = self.get_extended_attention_mask(attention_mask)
        for layer in self.encoder.layer:
            txt_embeds = layer(txt_embeds, attn_txt_mask)

        return txt_embeds, attn_txt_mask
