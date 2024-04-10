import numpy as np
import torch
from util import txt2embed, get_sincos_size_embed


def crop_resize(x, coord, pix_size, crop_size):
    s, h, w = x.shape
    ts, th, tw = crop_size

    sl = coord[1] - coord[0]
    hl = coord[3] - coord[2]
    wl = coord[5] - coord[4]

    cs = ts
    ch = th
    cw = tw

    if sl < cs:
        ms = (cs - sl) // 2
        ss = max(0, coord[0] - ms)
        se = min(s, coord[1] + ms)
    else:
        ss = coord[0]
        se = coord[1]

    if hl < ch:
        mh = (ch - hl) // 2
        hs = max(0, coord[2] - mh)
        he = min(h, coord[3] + mh)
    else:
        hs = coord[2]
        he = coord[3]

    if wl < cw:
        mw = (cw - wl) // 2
        ws = max(0, coord[4] - mw)
        we = min(w, coord[5] + mw)
    else:
        ws = coord[4]
        we = coord[5]

    x = x[ss:se, hs:he, ws:we]

    ps = pix_size[0] * cs / ts
    ph = pix_size[1] * ch / th
    pw = pix_size[2] * cw / tw

    x = torch.nn.functional.interpolate(
        torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(ts, th, tw),
        mode="trilinear",
        align_corners=False,
    ).numpy().squeeze()

    return x, [ps, ph, pw]


def normalize(x, data_min, data_max):
    x = [np.clip(xi, data_min, data_max) for xi in x]
    x = [(xi - data_min) / (data_max - data_min) for xi in x]
    x = [xi * 2 - 1 for xi in x]
    return x


def to_tensor(x):
    assert isinstance(x, list)
    return [torch.from_numpy(xi.copy()).unsqueeze(0) for xi in x]


def get_data(input_dict, args):

    pix_size = input_dict['pixel_size']
    ct_path = input_dict['ct_path']
    data_name = args.data_name
    coords = input_dict['coords'][data_name[0]]
    crop_size = args.crop_size
    patch_size = args.cube_size
    hu_range = args.hu_range
    embed_dim = args.embed_dim
    question = input_dict['question']
    clinical_txt = input_dict['clinical_txt']

    data_ori = np.load(ct_path)
    data, pix_size = crop_resize(data_ori, coords, pix_size, crop_size)

    data = normalize([data], hu_range[0], hu_range[1])[0]
    data = to_tensor([data])[0]

    sizes = np.array([[pix_size[0] * patch_size[0], pix_size[1] * patch_size[1], pix_size[2] * patch_size[2]]])
    size_embed = get_sincos_size_embed(embed_dim, sizes)
    size_embed = torch.from_numpy(size_embed).to(torch.float32)

    question_embed = txt2embed([question])
    question_ids = torch.LongTensor(question_embed['input_ids'])
    question_masks = torch.LongTensor(question_embed['attention_mask'])

    txt_embed = txt2embed([clinical_txt], max_length=160)
    txt_ids = torch.LongTensor(txt_embed['input_ids'])
    txt_masks = torch.LongTensor(txt_embed['attention_mask'])
    data_dict = {'data': data.unsqueeze(0), 'questions': question,
                 'questions_ids': question_ids.unsqueeze(0), 'questions_mask': question_masks.unsqueeze(0),
                 'data_size': torch.LongTensor(crop_size).unsqueeze(0),
                 'txt_ids': txt_ids.unsqueeze(0), 'txt_mask': txt_masks.unsqueeze(0),
                 'size_embed': size_embed.unsqueeze(0), "clinical_txt": clinical_txt,
                 'patch_size': patch_size,
                 'data_name': [data_name]}

    return data_dict
