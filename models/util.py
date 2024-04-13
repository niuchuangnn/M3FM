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


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def get_img_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:

    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -10000.0

    return attention_mask
