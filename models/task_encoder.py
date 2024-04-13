import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from models.util import Dropout, DropPathV2, MlpDrop


class QuestionAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_task,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim_task, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = Dropout()

    def forward(self, x, mask=None, drop_prob=0.0, attn_hook=None, require_attn_grad=False):
        # x: B x N x C
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)

        if attn_hook is not None:
            if require_attn_grad:
                attn.retain_grad()
            attn_hook(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x, drop_prob)

        return x


class QuestionBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
            self,
            dim,
            dim_task,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_func=QuestionAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            dim_task,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPathV2()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpDrop(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

        self.attn_prob = None

    def set_attn(self, attn_prob):
        self.attn_prob = attn_prob

    def forward(self, x, mask=None, drop_path_prob=0.0, drop_prob=0.0, attn_hook=False, require_attn_grad=False):
        # x_img: N x N1 x C
        # x_txt: N x N2 x C
        # t: N x 1 x C_t

        shortcut = x
        x = self.norm1(x)

        if attn_hook:
            x = self.attn(x, mask, drop_prob=drop_prob, attn_hook=self.set_attn, require_attn_grad=require_attn_grad)
        else:
            x = self.attn(x, mask, drop_prob=drop_prob)

        x = shortcut + self.drop_path(x, drop_path_prob)
        x = x + self.drop_path(self.mlp(self.norm2(x), drop_prob), drop_path_prob)

        return x


class TaskEncoder(nn.Module):
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
            self.trunc_init = trunc_init
            self.blocks = []
            for i in range(depth):
                block = QuestionBlock(
                    embed_dim,
                    prompt_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                if use_act_checkpoint:
                    block = checkpoint_wrapper(block)
                self.blocks.append(block)

            self.blocks = nn.ModuleList(self.blocks)
            self.norm = norm_layer(embed_dim)
            self.initialize_weights()

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
            num_blks = len(self.blocks)
            assert self.depth == num_blks
            dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]

            for i in range(num_blks):
                blk = self.blocks[i]
                x = blk(x, mask, drop_path_prob=dpr[i], drop_prob=drop,
                        attn_hook=attn_hook, require_attn_grad=require_attn_grad)

            x = self.norm(x)

        return x
