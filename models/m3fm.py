from functools import partial
import torch
import models.vit_window as vit_window
from torch import nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from transformers import RobertaModel
from transformers.activations import ACT2FN
import torch.nn.functional as F
from models.tokennizer_img import ImgTokenizerMSShare


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class PredictionHead(nn.Module):
    def __init__(self, in_size, hidden_size, hidden_act='gelu', layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(in_size, hidden_size)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, hidden_act='gelu', layer_norm_eps=1e-12, weight=None):
        super().__init__()
        self.transform = PredictionHead(in_size, hidden_size, hidden_act, layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class EncoderViT(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            no_qkv_bias=False,
            trunc_init=False,
            use_act_checkpoint=False,
            # drop_path=0.0,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.blocks = []

        # dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.depth = depth

        for i in range(depth):
            block = vit_window.BlockDrop(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=not no_qkv_bias,
                qk_scale=None,
                norm_layer=norm_layer,
                # drop_path=dpr[i],
            )

            if use_act_checkpoint:
                block = checkpoint_wrapper(block)

            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()
        print("Encoder initialized")

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, window_size=None, window_block_indexes=None,
                spatial_size=None, cls_embed=None, attention_mask=None,
                drop_path=0.0, drop=0.0):

        # apply Transformer blocks
        num_blks = len(self.blocks)

        assert self.depth == num_blks
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]

        for i in range(num_blks):
            blk = self.blocks[i]
            if i in window_block_indexes:
                wsize = window_size
            else:
                wsize = None
            x = blk(x, window_size=wsize, spatial_size=spatial_size, cls_embed=cls_embed, attention_mask=attention_mask,
                    drop_path_prob=dpr[i], drop_prob=drop)

        x = self.norm(x)

        return x


class TaskAttention(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            prompt_dim=1024,
            depth=8,
            num_heads=16,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            no_qkv_bias=False,
            trunc_init=False,
            use_act_checkpoint=False,
            # drop_path=0.0,
    ):
        super().__init__()
        self.depth = depth
        if self.depth > 0:
            # embed tokens
            # self.embed = nn.Linear(embed_dim, embed_dim, bias=True)
            self.trunc_init = trunc_init
            self.blocks = []
            for i in range(depth):
                block = vit_window.QuestionBlock(
                    embed_dim,
                    prompt_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    # drop_path=dpr[i],
                )
                if use_act_checkpoint:
                    block = checkpoint_wrapper(block)
                self.blocks.append(block)

            self.blocks = nn.ModuleList(self.blocks)
            self.norm = norm_layer(embed_dim)
            self.initialize_weights()
            print("Task Attention initialized")

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None, drop_path=0.0, drop=0.0, attn_hook=False, require_attn_grad=False):

        if self.depth > 0:
            # x = self.embed(x)
            num_blks = len(self.blocks)
            assert self.depth == num_blks
            dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]

            for i in range(num_blks):
                blk = self.blocks[i]
                x = blk(x, mask, drop_path_prob=dpr[i], drop_prob=drop,
                        attn_hook=attn_hook, require_attn_grad=require_attn_grad)

            x = self.norm(x)

        return x


class Hazard_Layer(nn.Module):
    def __init__(self, max_followup):
        super(Hazard_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def forward(self, x):
        B, D = x.size()  # hazards is (B, T)
        x1 = x[:, 0:-1]
        base_hazard = x[:, D - 1:D]
        T = D - 1

        expanded_hazards = x1.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
                expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        # base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob


def get_extended_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class MTHeadV1(nn.Module):
    def __init__(
            self,
            head_dim_dict,
            task_dim_dict,
            num_head_dict,
            task_head_map_dict,
            ta_embed_dim,
            loss_weight_dict,
            task_base_dict,
            task_enforce,
            max_followup=6,
    ):
        super().__init__()
        assert head_dim_dict is not None
        self.head_dim_dict = head_dim_dict
        assert task_dim_dict is not None
        self.task_dim_dict = task_dim_dict
        assert num_head_dict is not None
        self.num_head_dict = num_head_dict
        assert task_head_map_dict is not None
        self.task_head_map_dict = task_head_map_dict

        for kh in self.head_dim_dict.keys():
            hidden_size_kh = self.head_dim_dict[kh]
            out_size_kh = self.task_dim_dict[kh]
            num_head = self.num_head_dict[kh]
            for n in range(num_head):
                head_k = MLMHead(ta_embed_dim, hidden_size_kh, out_size_kh)
                head_k.apply(init_weights)
                head_name = '{}_{}'.format(kh, n)
                self.__setattr__(head_name, head_k)

        self.hazard_layer = Hazard_Layer(max_followup)

        assert loss_weight_dict is not None
        self.loss_weight_dict = loss_weight_dict
        assert task_base_dict is not None
        self.task_base_dict = task_base_dict
        self.task_enforce = task_enforce

    def forward(self, embeds, data_dict=None):
        logits = self.pred_logits(embeds)
        if self.training:
            return self.loss(logits, data_dict)
        else:
            return self.inference(logits)

    def pred_logits(self, x):
        out_logits_dict = {}
        for kt, kh in self.task_head_map_dict.items():
            num_head = self.num_head_dict[kh]
            logits_kt_all = []
            for n in range(num_head):
                head_name = '{}_{}'.format(kh, n)
                head_kh = self.__getattr__(head_name)
                out_kt = head_kh(x)
                if kt == 'cancer_risk':
                    out_kt = self.hazard_layer(out_kt)
                logits_kt_all.append(out_kt.unsqueeze(dim=1))
            logits_kt_all = torch.cat(logits_kt_all, dim=1)
            out_logits_dict[kt] = logits_kt_all
        return out_logits_dict

    def get_loss_weight(self, k, loss, loss_weight):
        if self.task_base_dict[k] > 0:
            loss_base = self.task_base_dict[k]
            loss_weight = (loss / (loss_base + 1e-12)) ** self.task_enforce
        return loss_weight

    def loss(self, logits, data_dict):
        answer_dict = data_dict['answers_dict']

        loss_dict = {}
        task_weight_dict = {}
        loss_all = 0.0
        keys_task = answer_dict.keys()
        num_loss = 0
        for k in keys_task:
            if torch.all(answer_dict[k] < 0):
                targets = torch.zeros_like(answer_dict[k])
                loss_weight = 0.0
            else:
                targets = answer_dict[k]
                loss_weight = 1.0
                num_loss = num_loss + 1

            task_logit = logits[k]
            num_head = task_logit.shape[1]

            targets = targets.unsqueeze(dim=1).repeat(1, num_head, 1)

            task_logit = task_logit.reshape([task_logit.shape[0] * num_head, task_logit.shape[2]])

            if k == 'cancer_risk':
                targets = targets.reshape([targets.shape[0] * num_head, targets.shape[2]])
                label_mask = data_dict['label_mask']
                label_mask = label_mask.unsqueeze(dim=1).repeat(1, num_head, 1)
                label_mask = label_mask.reshape([task_logit.shape[0], targets.shape[-1]])
                label_mask_norm = max(1e-8, torch.sum(label_mask))
                loss_k = F.binary_cross_entropy_with_logits(task_logit, targets, weight=label_mask,
                                                            reduction='sum') / label_mask_norm
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight
            else:
                targets = targets.reshape([targets.shape[0] * num_head, ])

                loss_k = F.cross_entropy(
                    task_logit,
                    targets,
                    weight=torch.FloatTensor(self.loss_weight_dict[k]).to(task_logit),
                    ignore_index=-100,
                )
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            loss_dict[k] = loss_k.unsqueeze(0)
            task_weight_dict[k] = loss_weight

        loss_all = loss_all / num_loss

        return loss_all, loss_dict, task_weight_dict

    def inference(self, logits):
        keys_task = logits.keys()
        output_dict = {}
        for k in keys_task:
            task_logit = logits[k]
            if k == 'cancer_risk':
                output_dict[k] = F.sigmoid(task_logit)
            else:
                output_dict[k] = F.softmax(task_logit, dim=2)
        return output_dict


class MTHeadV2(nn.Module):
    def __init__(
            self,
            head_dim_dict,
            task_dim_dict,
            num_head_dict,
            task_head_map_dict,
            ta_embed_dim,
            loss_weight_dict,
            task_base_dict,
            task_enforce,
            max_followup=6,
    ):
        super().__init__()
        assert head_dim_dict is not None
        self.head_dim_dict = head_dim_dict
        assert task_dim_dict is not None
        self.task_dim_dict = task_dim_dict
        assert num_head_dict is not None
        self.num_head_dict = num_head_dict
        assert task_head_map_dict is not None
        self.task_head_map_dict = task_head_map_dict

        self.head_task_map_dict = self.get_head_task_map()

        self.head_task_dim_dict, self.head_out_dim_dict = self.get_head_task_param()

        for kh in self.head_dim_dict.keys():
            hidden_size_kh = self.head_dim_dict[kh]
            out_size_kh = self.head_out_dim_dict[kh]
            head_k = MLMHead(ta_embed_dim, hidden_size_kh, out_size_kh)
            head_k.apply(init_weights)
            self.__setattr__(kh, head_k)

        self.hazard_layer = Hazard_Layer(max_followup)

        assert loss_weight_dict is not None
        self.loss_weight_dict = loss_weight_dict
        assert task_base_dict is not None
        self.task_base_dict = task_base_dict
        self.task_enforce = task_enforce

    def get_head_task_map(self):
        head_task_map_dict = {}
        for kt, kh in self.task_head_map_dict.items():
            if kh not in head_task_map_dict.keys():
                head_task_map_dict[kh] = [kt]
            else:
                head_task_map_dict[kh].append(kt)
        return head_task_map_dict

    def get_head_task_param(self):
        head_task_dim_dict = {}
        head_out_dim_dict = {}

        for kh, ktl in self.head_task_map_dict.items():
            dim_current = 0
            for k in ktl:
                num_head_k = self.num_head_dict[k]
                dim_k = self.task_dim_dict[k]
                head_task_dim_dict[k] = [dim_current, dim_current + dim_k * num_head_k]
                dim_current = dim_current + dim_k * num_head_k
            head_out_dim_dict[kh] = dim_current

        return head_task_dim_dict, head_out_dim_dict

    def forward(self, embeds, data_dict=None):
        logits = self.pred_logits(embeds)
        if self.training:
            return self.loss(logits, data_dict)
        else:
            return self.inference(logits)

    def pred_logits(self, x):
        out_logits_dict = {}

        for kh, ktl in self.head_task_map_dict.items():
            head_kh = self.__getattr__(kh)
            out_kh = head_kh(x)

            for kt in ktl:
                num_head = self.num_head_dict[kt]
                dim_s, dim_e = self.head_task_dim_dict[kt]
                out_kt = out_kh[:, dim_s:dim_e]
                B = out_kt.shape[0]
                out_kt = out_kt.reshape([B, num_head, -1])
                if kt == 'cancer_risk':
                    out_kt = out_kt.reshape([B * num_head, -1])
                    out_kt = self.hazard_layer(out_kt)
                    out_kt = out_kt.reshape([B, num_head, -1])

                out_logits_dict[kt] = out_kt

        return out_logits_dict

    def get_loss_weight(self, k, loss, loss_weight):
        if self.task_base_dict[k] > 0:
            loss_base = self.task_base_dict[k]
            loss_weight = (loss / (loss_base + 1e-12)) ** self.task_enforce
        return loss_weight

    def loss(self, logits, data_dict):
        answer_dict = data_dict['answers_dict']

        loss_dict = {}
        task_weight_dict = {}
        loss_all = 0.0
        keys_task = answer_dict.keys()
        num_loss = 0
        for k in keys_task:
            if torch.all(answer_dict[k] < 0):
                targets = torch.zeros_like(answer_dict[k])
                loss_weight = 0.0
            else:
                targets = answer_dict[k]
                loss_weight = 1.0
                num_loss = num_loss + 1

            task_logit = logits[k]
            num_head = task_logit.shape[1]

            targets = targets.unsqueeze(dim=1).repeat(1, num_head, 1)

            task_logit = task_logit.reshape([task_logit.shape[0] * num_head, task_logit.shape[2]])

            if k == 'cancer_risk':
                targets = targets.reshape([targets.shape[0] * num_head, targets.shape[2]])
                label_mask = data_dict['label_mask']
                label_mask = label_mask.unsqueeze(dim=1).repeat(1, num_head, 1)
                label_mask = label_mask.reshape([task_logit.shape[0], targets.shape[-1]])
                label_mask_norm = max(1e-8, torch.sum(label_mask))
                loss_k = F.binary_cross_entropy_with_logits(task_logit, targets, weight=label_mask,
                                                            reduction='sum') / label_mask_norm
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            elif k == 'clinical_number':
                task_logit = task_logit.reshape(task_logit.shape[0], 10, -1)
                # num_digit = task_logit.shape[2]
                targets = targets.reshape([targets.shape[0] * num_head, targets.shape[2]])
                loss_k = F.cross_entropy(
                    task_logit,
                    targets,
                    ignore_index=-100,
                )
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            else:
                targets = targets.reshape([targets.shape[0] * num_head, ])

                loss_k = F.cross_entropy(
                    task_logit,
                    targets,
                    weight=torch.FloatTensor(self.loss_weight_dict[k]).to(task_logit),
                    ignore_index=-100,
                )
                loss_weight = self.get_loss_weight(k, loss_k.item(), loss_weight)
                loss_all = loss_all + loss_k * loss_weight

            loss_dict[k] = loss_k.unsqueeze(0)
            task_weight_dict[k] = loss_weight

        loss_all = loss_all / num_loss

        return loss_all, loss_dict, task_weight_dict

    def inference(self, logits):
        keys_task = logits.keys()
        output_dict = {}
        for k in keys_task:
            task_logit = logits[k]
            if k == 'cancer_risk':
                output_dict[k] = F.sigmoid(task_logit)
            elif k == 'clinical_number':
                task_logit = task_logit.reshape(task_logit.shape[0], task_logit.shape[1], 10, -1)
                task_logit = F.softmax(task_logit, dim=2)
                task_out = task_logit.argmax(dim=2)
                task_out = task_out[:, :, 0:1] * 100 + task_out[:, :, 1:2] * 10 + task_out[:, :, 2:3]
                output_dict[k] = task_out.contiguous()
            else:
                output_dict[k] = F.softmax(task_logit, dim=2)
        return output_dict


class M3FM(nn.Module):
    def __init__(
            self,
            img_size_list=None,
            in_chans=1,
            patch_size_list=None,
            window_size=(1, 4, 4),
            window_block_indexes=(),
            embed_dim_img=1024,
            depth=24,
            num_heads=16,
            ta_embed_dim=1024,
            ta_prompt_dim=1024,
            ta_depth=4,
            ta_num_heads=16,
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            no_qkv_bias=False,
            sep_pos_embed=True,
            trunc_init=False,
            use_act_checkpoint=True,
            cls_embed_img=True,
            txt_embed_dim=768,
            loss_weight_dict=None,
            max_followup=6,
            task_base_dict=None,
            task_enforce=0.1,
            drop_dict=None,
            drop_dict_ta=None,
            drop_path_dict=None,
            drop_path_dict_ta=None,
            modalities=['img', 'txt', 'task'],
            head_dim_dict=None,
            task_dim_dict=None,
            num_head_dict=None,
            multi_task_head='v1',
            task_head_map_dict=None,
            num_task_registers=3,
            **kwargs,
    ):
        super().__init__()
        self.modalities = modalities
        if 'img' in modalities:
            self.img_tokenizer = ImgTokenizerMSShare(
                img_size_list,
                patch_size_list,
                in_chans,
                embed_dim_img,
                sep_pos_embed=sep_pos_embed,
                trunc_init=trunc_init,
                cls_embed=cls_embed_img,
                use_tokenizer_all=True,
            )

            self.encoder_img = EncoderViT(
                embed_dim=embed_dim_img,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                no_qkv_bias=no_qkv_bias,
                trunc_init=trunc_init,
                use_act_checkpoint=use_act_checkpoint,
            )

            self.cross_embedding_img = nn.Linear(embed_dim_img, ta_embed_dim)
            self.cross_embedding_img.apply(init_weights)

            self.embed_dim_img = embed_dim_img
            self.window_size = window_size
            self.window_block_indexes = window_block_indexes

            self.cls_embed_img = cls_embed_img
            self.ims = None

        if 'txt' in modalities or 'task' in modalities:
            self.encoder_txt = RobertaModel.from_pretrained('demo_data/pretrained_models/roberta-base')
            self.encoder_txt.pooler = None
            self.cross_embedding_txt = nn.Linear(txt_embed_dim, ta_embed_dim)
            self.cross_embedding_txt.apply(init_weights)

        if 'task' in modalities:
            self.num_task_registers = num_task_registers
            if num_task_registers is not None:
                self.task_registers = nn.Parameter(torch.zeros(1, num_task_registers, ta_embed_dim))
                torch.nn.init.trunc_normal_(self.task_registers, std=0.02)
            else:
                self.task_registers = None

            self.task_attention = TaskAttention(embed_dim=ta_embed_dim,
                                                prompt_dim=ta_prompt_dim,
                                                depth=ta_depth,
                                                num_heads=ta_num_heads,
                                                mlp_ratio=mlp_ratio,
                                                norm_layer=norm_layer,
                                                no_qkv_bias=no_qkv_bias,
                                                trunc_init=trunc_init,
                                                use_act_checkpoint=use_act_checkpoint,
                                                )
            self.cross_embedding_task = nn.Linear(txt_embed_dim, ta_prompt_dim)
            self.cross_embedding_task.apply(init_weights)

        if multi_task_head == 'v1':
            self.multi_task_head = MTHeadV1(head_dim_dict,
                                            task_dim_dict,
                                            num_head_dict,
                                            task_head_map_dict,
                                            ta_embed_dim,
                                            loss_weight_dict,
                                            task_base_dict,
                                            task_enforce,
                                            max_followup,
                                            )
        elif multi_task_head == 'v2':
            self.multi_task_head = MTHeadV2(head_dim_dict,
                                            task_dim_dict,
                                            num_head_dict,
                                            task_head_map_dict,
                                            ta_embed_dim,
                                            loss_weight_dict,
                                            task_base_dict,
                                            task_enforce,
                                            max_followup,
                                            )
        else:
            raise TypeError

        self.drop_dict = drop_dict
        self.drop_dict_ta = drop_dict_ta
        self.drop_path_dict = drop_path_dict
        self.drop_path_dict_ta = drop_path_dict_ta

    def load_data(self, data_dict):

        if 'data' in data_dict.keys():
            data = data_dict['data'].to(torch.float32).cuda(data_dict['gpu'], non_blocking=True)
            data = data.reshape([data.shape[0] * data.shape[1], 1, data.shape[2], data.shape[3], data.shape[4]])
            data_dict['data'] = data

        if 'size_embed' in data_dict.keys():
            size_embed = data_dict['size_embed'].to(torch.float32).cuda(data_dict['gpu'], non_blocking=True)
            size_embed = size_embed.reshape(size_embed.shape[0] * size_embed.shape[1], 1, size_embed.shape[-1])
            data_dict['size_embed'] = size_embed

        if 'txt_ids' in data_dict.keys():
            txt_ids = data_dict['txt_ids'].to(torch.long).cuda(data_dict['gpu'], non_blocking=True)
            txt_ids = txt_ids.reshape(txt_ids.shape[0] * txt_ids.shape[1], txt_ids.shape[-1])
            data_dict['txt_ids'] = txt_ids
            txt_mask = data_dict['txt_mask'].to(torch.long).cuda(data_dict['gpu'], non_blocking=True)
            txt_mask = txt_mask.reshape(txt_ids.shape)
            data_dict['txt_mask'] = txt_mask

        if 'questions_ids' in data_dict.keys():
            questions_ids = data_dict['questions_ids'].to(torch.long).cuda(data_dict['gpu'], non_blocking=True)
            questions_ids = questions_ids.reshape(questions_ids.shape[0] * questions_ids.shape[1],
                                                  questions_ids.shape[-1])
            data_dict['questions_ids'] = questions_ids
            questions_mask = data_dict['questions_mask'].to(torch.long).cuda(data_dict['gpu'], non_blocking=True)
            questions_mask = questions_mask.reshape(questions_ids.shape)
            data_dict['questions_mask'] = questions_mask

        if 'label_mask' in data_dict.keys():
            label_mask = data_dict['label_mask'].to(torch.float32).cuda(data_dict['gpu'], non_blocking=True)
            label_mask = label_mask.reshape(label_mask.shape[0] * label_mask.shape[1], label_mask.shape[-1])
            data_dict['label_mask'] = label_mask

        if 'answers_dict' in data_dict.keys():
            for k, v in data_dict['answers_dict'].items():
                v = v.cuda(data_dict['gpu'], non_blocking=True)
                v = v.reshape(v.shape[0] * v.shape[1], v.shape[-1])
                data_dict['answers_dict'][k] = v

        if 'att_mask' in data_dict.keys():
            att_mask = data_dict['att_mask'].to(torch.long).cuda(data_dict['gpu'], non_blocking=True)
            att_mask = att_mask.reshape(att_mask.shape[0] * att_mask.shape[1], att_mask.shape[-1])
            data_dict['att_mask'] = att_mask

        data_name_all = data_dict['data_name']
        data_name = []
        for dn in data_name_all:
            data_name = data_name + dn
        data_dict['data_name'] = data_name

    def get_drop(self, name):
        if self.drop_dict is not None:
            drop = self.drop_dict[name]
        else:
            drop = 0.0

        if self.drop_dict_ta is not None:
            drop_ta = self.drop_dict_ta[name]
        else:
            drop_ta = 0.0

        if self.drop_path_dict is not None:
            drop_path = self.drop_path_dict[name]
        else:
            drop_path = 0.0

        if self.drop_path_dict_ta is not None:
            drop_path_ta = self.drop_path_dict_ta[name]
        else:
            drop_path_ta = 0.0

        return drop, drop_ta, drop_path, drop_path_ta

    def forward(self, data_dict):
        self.load_data(data_dict)
        embeds = self.pred_embeds(data_dict)
        out = self.multi_task_head(embeds, data_dict)
        return out

    def pred_embeds(self, data_dict):
        drop, drop_ta, drop_path, drop_path_ta = self.get_drop(data_dict['data_name'][0])

        if 'attn_hook' in data_dict.keys():
            attn_hook = data_dict['attn_hook']
            require_attn_grad = data_dict['require_attn_grad']
        else:
            attn_hook = False
            require_attn_grad = False

        embeds_inputs = []
        masks_all = []

        if 'img' in self.modalities:
            if 'att_mask' in data_dict.keys():
                attn_img_mask = data_dict['att_mask'].unsqueeze(1).unsqueeze(1)
            else:
                attn_img_mask = None

            if 'data' in data_dict.keys():
                imgs = data_dict['data']
                if 'size_embed' in data_dict.keys():
                    size_embed = data_dict['size_embed']
                else:
                    size_embed = None
            else:
                imgs = torch.zeros((1, 1, 4, 4, 4)).to(torch.float32).cuda(data_dict['gpu'], non_blocking=True)
                size_embed = None

            self.ims = (imgs.shape[2], imgs.shape[3], imgs.shape[4])
            img_embeds = self.img_tokenizer(imgs)

            if isinstance(size_embed, torch.Tensor):
                img_embeds = img_embeds + size_embed

            if 'data' not in data_dict.keys():
                assert 'txt_ids' in data_dict.keys()
                img_embeds = (torch.zeros((data_dict['txt_ids'].shape[0], 9, self.embed_dim_img)).to(
                    torch.float32).cuda(data_dict['gpu'], non_blocking=True) + img_embeds) * 0.0
                input_size = (2, 2, 2)
                self.window_size = None
                self.window_block_indexes = ()
            else:
                input_size = self.img_tokenizer.__getattr__('tokenizer_{}'.format(self.ims)).input_size

            img_embeds = self.encoder_img(img_embeds,
                                          window_size=self.window_size,
                                          window_block_indexes=self.window_block_indexes,
                                          spatial_size=input_size,
                                          cls_embed=self.cls_embed_img,
                                          attention_mask=attn_img_mask,
                                          drop_path=drop_path,
                                          drop=drop)

            img_embeds = self.cross_embedding_img(img_embeds)

            embeds_inputs.append(img_embeds)

            if 'data' not in data_dict.keys():
                attn_img_mask = torch.zeros(img_embeds.shape[0], img_embeds.shape[1]).to(img_embeds.device)
                attn_img_mask = get_extended_attention_mask(attn_img_mask)

            if attn_img_mask is None:
                attn_img_mask = torch.ones(img_embeds.shape[0], img_embeds.shape[1]).to(img_embeds.device)
                attn_img_mask = get_extended_attention_mask(attn_img_mask)

            masks_all.append(attn_img_mask)

        if 'txt' in self.modalities:
            txt_ids = data_dict['txt_ids']
            text_masks = data_dict['txt_mask']

            txt_embeds = self.encoder_txt.embeddings(input_ids=txt_ids)
            txt_input_shape = text_masks.size()
            attn_txt_mask = self.encoder_txt.get_extended_attention_mask(text_masks, txt_input_shape,
                                                                         txt_embeds.device)
            for layer in self.encoder_txt.encoder.layer:
                txt_embeds = layer(txt_embeds, attn_txt_mask)[0]

            txt_embeds = self.cross_embedding_txt(txt_embeds)

            embeds_inputs.append(txt_embeds)
            masks_all.append(attn_txt_mask)

        embeds_all = torch.cat(embeds_inputs, dim=1)
        masks_all = torch.cat(masks_all, dim=-1)

        if 'task' in self.modalities:
            if self.task_registers is not None:
                task_registers = self.task_registers.expand(embeds_all.shape[0], -1, -1)
                task_registers_mask = torch.zeros(task_registers.shape[0], 1, 1, task_registers.shape[1]).to(
                    task_registers)
                embeds_all = torch.cat([task_registers, embeds_all], dim=1)
                masks_all = torch.cat([task_registers_mask, masks_all], dim=-1)

            task_ids = data_dict['questions_ids']
            task_masks = data_dict['questions_mask']

            task_embeds = self.encoder_txt.embeddings(input_ids=task_ids)
            task_input_shape = task_masks.size()
            extend_task_masks = self.encoder_txt.get_extended_attention_mask(task_masks, task_input_shape,
                                                                             task_embeds.device)
            for layer in self.encoder_txt.encoder.layer:
                task_embeds = layer(task_embeds, extend_task_masks)[0]

            task_embeds = task_embeds[:, 0, ...]
            task_embeds = self.cross_embedding_task(task_embeds)
            task_embeds = task_embeds.unsqueeze(1).expand(embeds_all.shape[0], -1, -1)
            task_embeds_mask = torch.zeros(task_embeds.shape[0], 1, 1, 1).to(task_embeds)
            embeds_all = torch.cat([task_embeds, embeds_all], dim=1)
            masks_all = torch.cat([task_embeds_mask, masks_all], dim=-1)
            embeds_all = self.task_attention(embeds_all, masks_all, drop_path=drop_path_ta, drop=drop_ta,
                                             attn_hook=attn_hook, require_attn_grad=require_attn_grad)

        embeds_pred = embeds_all[:, 0, :]

        return embeds_pred


def m3fm_base(**kwargs):
    window_blocs_indexes = []  # list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
    window_blocs_indexes_ta = []
    window_blocs_indexes_de = []
    model = M3FM(
        img_size=(16, 512, 512),
        in_chans=1,
        patch_size=(4, 16, 16),
        window_size=(1, 4, 4),
        window_block_indexes=(),
        embed_dim_img=768,
        depth=12,
        num_heads=12,
        ta_embed_dim=768,
        ta_prompt_dim=768,
        ta_depth=4,
        ta_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        use_act_checkpoint=True,
        drop_path=0.0,
        drop_path_ta=0.0,
        cls_embed_img=True,
        txt_embed_dim=768,
        **kwargs,
    )
    return model


def m3fm_base1024(**kwargs):
    window_blocs_indexes = []  # list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
    window_blocs_indexes_ta = []
    window_blocs_indexes_de = []
    model = M3FM(
        img_size=(16, 512, 512),
        in_chans=1,
        patch_size=(4, 16, 16),
        window_size=(1, 4, 4),
        window_block_indexes=(),
        embed_dim_img=1024,
        depth=12,
        num_heads=16,
        ta_embed_dim=1024,
        ta_prompt_dim=1024,
        ta_depth=4,
        ta_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        # use_act_checkpoint=True,
        drop_path=0.0,
        drop_path_ta=0.0,
        cls_embed_img=True,
        txt_embed_dim=768,
        **kwargs,
    )
    return model


def m3fm_large(**kwargs):
    window_blocs_indexes = []  # list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
    window_blocs_indexes_ta = []
    window_blocs_indexes_de = []
    model = M3FM(
        img_size=(16, 512, 512),
        in_chans=1,
        patch_size=(4, 16, 16),
        window_size=(1, 4, 4),
        window_block_indexes=(),
        embed_dim_img=1024,
        depth=24,
        num_heads=16,
        ta_embed_dim=1024,
        ta_prompt_dim=1024,
        ta_depth=4,
        ta_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        # use_act_checkpoint=True,
        # drop_path=0.0,
        # drop_path_ta=0.0,
        cls_embed_img=True,
        txt_embed_dim=768,
        **kwargs,
    )
    return model


def m3fm_huge(**kwargs):
    window_blocs_indexes = []  # list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
    window_blocs_indexes_ta = []
    window_blocs_indexes_de = []
    model = M3FM(
        img_size=(16, 512, 512),
        in_chans=1,
        patch_size=(4, 16, 16),
        window_size=(1, 4, 4),
        window_block_indexes=(),
        embed_dim_img=1280,
        depth=32,
        num_heads=16,
        ta_embed_dim=1280,
        ta_prompt_dim=1280,
        ta_depth=4,
        ta_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        # use_act_checkpoint=True,
        drop_path=0.0,
        drop_path_ta=0.0,
        cls_embed_img=True,
        txt_embed_dim=768,
        **kwargs,
    )
    return model
