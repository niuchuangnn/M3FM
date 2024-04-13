import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from models.util import Dropout, DropPathV2, MlpDrop, window_partition_3d, window_partition_2d, window_unpartition_2d, window_unpartition_3d


class AttentionDrop(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        # proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = Dropout()

    def forward(self, x, attention_mask=None, drop_prob=0.0):
        if len(x.shape) == 5:
            B, ws, wh, ww, C = x.shape
            N = ws * wh * ww
        elif len(x.shape) == 4:
            B, wh, ww, C = x.shape
            N = wh * ww
        elif len(x.shape) == 3:
            B, N, C = x.shape
        else:
            raise TypeError

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

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, drop_prob)
        x = x.view(B, -1, C)
        return x


class BlockDrop(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            # drop=0.0,
            attn_drop=0.0,
            # drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_func=AttentionDrop,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            # proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPathV2()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpDrop(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            # drop=drop,
        )

    def forward(self, x, window_size=None, spatial_size=None, cls_embed=False, attention_mask=None,
                drop_path_prob=0.0, drop_prob=0.0):
        # x: N, s, h, w, C or N, (s*h*w)+1, C
        shortcut = x
        x = self.norm1(x)

        if cls_embed:
            assert spatial_size is not None
            N, L, C = x.shape
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
            x = x.reshape(N, *spatial_size, C)

        if window_size is not None:
            if len(x.shape) == 3:
                N, L, C = x.shape
                x = x.reshape(N, *spatial_size, C)
            if len(x.shape) == 5:
                x_size = x.shape[1], x.shape[2], x.shape[3]
                x, pad = window_partition_3d(x, window_size)
            elif len(x.shape) == 4:
                x_size = x.shape[1], x.shape[2]
                x, pad = window_partition_2d(x, window_size)
            else:
                raise TypeError

        if cls_embed:
            x = x.reshape([x.shape[0], -1, C])
            x = torch.cat([x_cls.expand(-1, x.shape[0]//x_cls.shape[0], C).reshape([-1, 1, C]), x], dim=1)

        x = self.attn(x, attention_mask, drop_prob)

        if cls_embed:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]

        if window_size is not None:
            if len(x_size) == 3:
                x = window_unpartition_3d(x, window_size, pad, x_size)
            else:
                x = window_unpartition_2d(x, window_size, pad, x_size)
        else:
            if len(x.shape) > 3:
                x = x.view(shortcut.shape)
            else:
                if cls_embed:
                    x = x.view(shortcut.shape[0], -1, shortcut.shape[2])
                else:
                    x = x.view(shortcut.shape)

        if cls_embed:
            x_cls = x_cls.reshape(x.shape[0], -1, C).mean(dim=1, keepdim=True)
            x = x.reshape([x.shape[0], -1, C])
            x = torch.cat([x_cls, x], dim=1)
        else:
            x = x.reshape([x.shape[0], -1, x.shape[-1]])

        x = shortcut + self.drop_path(x, drop_path_prob)

        x = x + self.drop_path(self.mlp(self.norm2(x), drop_prob), drop_path_prob)

        return x


class CTViT(nn.Module):
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
            block = BlockDrop(
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
