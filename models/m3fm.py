from functools import partial
import torch
from torch import nn
from models.ct_tokenizer import ImgTokenizerMSShare
from models.ctvit import CTViT
from models.text_transformer import TextTransformer, TextTransformerConfig
from models.task_encoder import TaskEncoder
from models.predictors import MTHead
from models.util import init_weights, get_img_attention_mask


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

            self.encoder_img = CTViT(
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
            # self.encoder_txt = RobertaModel.from_pretrained('demo_data/pretrained_models/roberta-base')
            # self.encoder_txt.pooler = None
            cfg = TextTransformerConfig()
            self.encoder_txt = TextTransformer(cfg)
            self.cross_embedding_txt = nn.Linear(txt_embed_dim, ta_embed_dim)
            self.cross_embedding_txt.apply(init_weights)

        if 'task' in modalities:
            self.num_task_registers = num_task_registers
            if num_task_registers is not None:
                self.task_registers = nn.Parameter(torch.zeros(1, num_task_registers, ta_embed_dim))
                torch.nn.init.trunc_normal_(self.task_registers, std=0.02)
            else:
                self.task_registers = None

            self.task_attention = TaskEncoder(embed_dim=ta_embed_dim,
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

        self.multi_task_head = MTHead(head_dim_dict,
                                      task_dim_dict,
                                      num_head_dict,
                                      task_head_map_dict,
                                      ta_embed_dim,
                                      loss_weight_dict,
                                      task_base_dict,
                                      task_enforce,
                                      max_followup,
                                      )

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
                attn_img_mask = get_img_attention_mask(attn_img_mask)

            if attn_img_mask is None:
                attn_img_mask = torch.ones(img_embeds.shape[0], img_embeds.shape[1]).to(img_embeds.device)
                attn_img_mask = get_img_attention_mask(attn_img_mask)

            masks_all.append(attn_img_mask)

        if 'txt' in self.modalities:
            txt_ids = data_dict['txt_ids']
            text_masks = data_dict['txt_mask']

            txt_embeds, attn_txt_mask = self.encoder_txt(txt_ids, text_masks)

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

            task_embeds, attn_task_mask = self.encoder_txt(task_ids, task_masks)

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
        drop_path=0.0,
        drop_path_ta=0.0,
        cls_embed_img=True,
        txt_embed_dim=768,
        **kwargs,
    )
    return model
