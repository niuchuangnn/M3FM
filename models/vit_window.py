import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x, p):
        return F.dropout(x, p, self.training, self.inplace)


class MlpDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = Dropout()

    def forward(self, x, drop_prob=0.0):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, drop_prob)
        x = self.fc2(x)
        x = self.drop(x, drop_prob)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPathV2(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self):
        super(DropPathV2, self).__init__()
        # self.drop_prob = drop_prob

    def forward(self, x, drop_prob):
        return drop_path(x, drop_prob, self.training)


def window_partition_3d(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, s, h, w, C].
        window_size (list): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_s, window_h, window_w, C].
        (sp, hp, wp): padded height and width before partition
    """
    B, s, h, w, C = x.shape

    pad_s = (window_size[0] - s % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - h % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - w % window_size[2]) % window_size[2]
    if pad_s > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_s, 0, pad_h, 0, pad_w))
    sp, hp, wp = s + pad_s, h + pad_h, w + pad_w

    x = x.view(B, sp//window_size[0], window_size[0], hp // window_size[1], window_size[1], wp // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows, (sp, hp, wp)


def window_unpartition_3d(windows, window_size, pad_shw, shw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_s, window_h, window_w, C].
        window_size (Tuple): window size.
        pad_shw (Tuple): padded slices, height and width (sp, hp, wp).
        shw (Tuple): original slices, height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, s, h, w, C].
    """
    sp, hp, wp = pad_shw
    s, h, w = shw
    B = windows.shape[0] // (sp * hp * wp // window_size[0] // window_size[1] // window_size[2])
    x = windows.view(B, sp // window_size[0], hp // window_size[1], wp // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, sp, hp, wp, -1)

    if sp > s or hp > h or wp > w:
        x = x[:, :s, :h, :w, :].contiguous()
    return x


def window_partition_2d(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size[0], window_size[0], Wp // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows, (Hp, Wp)


def window_unpartition_2d(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size[0] // window_size[1])
    x = windows.view(B, Hp // window_size[0], Wp // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


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