import torch
import torch.nn as nn


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep


class PatchEmbed3D(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        in_chans=1,
        embed_dim=1024,
    ):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        assert img_size[2] % patch_size[2] == 0

        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        self.input_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        # print(
        #     f"img_size {img_size} patch_size {patch_size}"
        # )
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = num_patches

        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1],
                          img_size[2] // patch_size[2])

        kernel_size = list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, S, H, W = x.shape
        assert (
            S == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), f"Input image size ({S}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).permute([0, 2, 3, 4, 1])
        return x


class PatchEmbed3DConv(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        in_chans=1,
        embed_dim=1024,
        conv=None,
    ):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        assert img_size[2] % patch_size[2] == 0

        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        self.input_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        # print(
        #     f"img_size {img_size} patch_size {patch_size}"
        # )
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = num_patches

        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1],
                          img_size[2] // patch_size[2])

        if conv is None:
            kernel_size = list(patch_size)
            self.proj = nn.Conv3d(
                in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
            )
        else:
            self.proj = conv

    def forward(self, x):
        B, C, S, H, W = x.shape
        assert (
            S == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2]
        ), f"Input image size ({S}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).permute([0, 2, 3, 4, 1])
        return x


class ImgTokenizer3D(nn.Module):
    def __init__(
            self,
            img_size=(32, 64, 64),
            patch_size=(8, 16, 16),
            in_chans=1,
            embed_dim=1024,
            sep_pos_embed=True,
            trunc_init=False,
            cls_embed=False,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.img_size = img_size
        self.cls_embed = cls_embed
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3D(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
        )

        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        self.num_patches = num_patches
        self.input_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.input_size[1] * self.input_size[2], embed_dim)
            )
            self.pos_embed_slice = nn.Parameter(
                torch.zeros(1, self.input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.initialize_weights()

        # print("Image Tokenizer initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_slice, std=0.02)
            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

    def forward(self, x, mask_ratio=0):
        # embed patches
        x = self.patch_embed(x)
        N, s, h, w, C = x.shape
        x = x.reshape(N, s * h * w, C)

        if mask_ratio > 0:
            x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_slice,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)

            if mask_ratio > 0:
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
                )

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if mask_ratio > 0:
                if self.cls_embed:
                    cls_ind = 1
                else:
                    cls_ind = 0
                pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
                )
                if self.cls_embed:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                            pos_embed,
                        ],
                        1,
                    )
            else:
                pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)

        x = x + pos_embed

        if mask_ratio > 0:
            return x, mask, ids_restore

        return x


class ImgTokenizer3DConv(nn.Module):
    def __init__(
            self,
            img_size=(32, 64, 64),
            patch_size=(8, 16, 16),
            in_chans=1,
            embed_dim=1024,
            sep_pos_embed=True,
            trunc_init=False,
            cls_embed=False,
            conv=None,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.img_size = img_size
        self.cls_embed = cls_embed
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed3DConv(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            conv
        )

        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (img_size[2] // patch_size[2])
        )
        self.num_patches = num_patches
        self.input_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.input_size[1] * self.input_size[2], embed_dim)
            )
            self.pos_embed_slice = nn.Parameter(
                torch.zeros(1, self.input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.initialize_weights()

        # print("Image Tokenizer initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_slice, std=0.02)
            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

    def forward(self, x, mask_ratio=0):
        # embed patches
        x = self.patch_embed(x)
        N, s, h, w, C = x.shape
        x = x.reshape(N, s * h * w, C)

        if mask_ratio > 0:
            x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_slice,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)

            if mask_ratio > 0:
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
                )

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if mask_ratio > 0:
                if self.cls_embed:
                    cls_ind = 1
                else:
                    cls_ind = 0
                pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
                )
                if self.cls_embed:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                            pos_embed,
                        ],
                        1,
                    )
            else:
                pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)

        x = x + pos_embed

        if mask_ratio > 0:
            return x, mask, ids_restore

        return x


class ImgTokenizerMSShare(nn.Module):
    def __init__(
            self,
            img_size_list=None,
            patch_size_list=None,
            in_chans=1,
            embed_dim=1024,
            sep_pos_embed=True,
            trunc_init=False,
            cls_embed=False,
            use_tokenizer_all=False,
    ):
        super().__init__()

        if img_size_list is None:
            img_size_list = [(16, 448, 320),
                             (128, 320, 448),
                             (128, 192, 224),
                             (128, 448, 320),
                             ]
        self.img_size_list = img_size_list

        if patch_size_list is None:
            patch_size_list = [(4, 16, 16),
                               (16, 16, 16),
                               (16, 16, 16),
                               (16, 16, 16),
                               ]
        self.patch_size_list = patch_size_list

        self.conv1 = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(4, 16, 16), stride=(4, 16, 16)
        )

        self.conv2 = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(16, 16, 16), stride=(16, 16, 16)
        )

        self.conv3 = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(16, 16, 16), stride=(16, 16, 16)
        )

        self.conv4 = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(16, 16, 16), stride=(16, 16, 16)
        )

        conv_list = [self.conv1, self.conv2, self.conv3, self.conv4]

        assert len(img_size_list) == len(patch_size_list)
        self.num_scale = len(img_size_list)

        for n in range(self.num_scale):
            img_size = img_size_list[n]
            patch_size = patch_size_list[n]
            name_n = 'tokenizer_{}'.format(img_size)
            tokenizer_n = ImgTokenizer3DConv(img_size, patch_size, in_chans,
                                             embed_dim, sep_pos_embed,
                                             trunc_init, cls_embed, conv_list[n])
            self.__setattr__(name_n, tokenizer_n)

        self.use_conv_all = use_tokenizer_all

    def forward(self, x, mask_ratio=0.0):
        
        img_size = (x.shape[2], x.shape[3], x.shape[4])

        if self.use_conv_all:
            x_rest = 0.0
            for n in range(self.num_scale):
                if img_size != self.img_size_list[n]:
                    name_nr = 'tokenizer_{}'.format(self.img_size_list[n])
                    tokenizer_nr = self.__getattr__(name_nr)
                    x_rest = x_rest + tokenizer_nr(torch.zeros(self.img_size_list[n]).unsqueeze(0).unsqueeze(0).to(x)).sum() * 0.0
        else:
            x_rest = 0.0

        if img_size in self.img_size_list:
            name_n = 'tokenizer_{}'.format(img_size)
            tokenizer_n = self.__getattr__(name_n)
            if mask_ratio > 0:
                x, mask, ids_restore = tokenizer_n(x, mask_ratio)
                return x+x_rest, mask, ids_restore
            else:
                x = tokenizer_n(x)
                return x + x_rest
        else:
            return x_rest
