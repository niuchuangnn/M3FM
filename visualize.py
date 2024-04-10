import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from util import embed2tokens
import matplotlib.colors as mcolors


def compute_attention_map(model, data_dict, task_layer_id,
                          task_token, register_token, img_cls_token, img_token, txt_token,
                          img_token_shape, img_size):
    with torch.no_grad():
        output_ori = model(data_dict)

    output = output_ori[data_dict['data_name'][0]]
    if len(output.shape) == 3:
        output = output.max(dim=1).values
    bs = output.shape[0]

    task_attn_blocks = list(model.task_attention.blocks)
    num_tokens = task_attn_blocks[0].attn_prob.shape[-1]

    R = torch.eye(num_tokens, num_tokens, dtype=task_attn_blocks[0].attn_prob.dtype).to(output.device)
    R = torch.cat([R.unsqueeze(0)]*bs, dim=0)

    for i, blk in enumerate(task_attn_blocks):

        if i != task_layer_id and task_layer_id != -1:
            continue

        attn = blk.attn_prob.detach()
        attn = attn.reshape(attn.shape[0], -1, attn.shape[-1], attn.shape[-1])

        attn = attn.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(attn, R)

    s_img = task_token + register_token + img_cls_token
    e_img = s_img + img_token

    image_relevance = R[:, 0, s_img:e_img]

    txt_relevance = R[:, 0, e_img:e_img+txt_token].cpu().numpy()

    image_relevance = image_relevance.reshape([bs] + img_token_shape).unsqueeze(1)

    image_relevance = torch.nn.functional.interpolate(image_relevance, size=img_size, mode='trilinear')
    image_relevance = image_relevance.squeeze().cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (
            image_relevance.max() - image_relevance.min() + 1e-32)

    return image_relevance, txt_relevance, output_ori


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def get_attention_map(img, image_relevance, args):
    if args.task_id == 15:
        image_relevance = 1 - image_relevance

    img_att = np.zeros(list(img.shape)+[3]).astype(np.uint8)
    for s in range(img_att.shape[0]):
        img_s = img[s]
        image_relevance_s = image_relevance[s]
        img_s = np.array([img_s, img_s, img_s]).transpose([1, 2, 0])
        img_s = (img_s - img_s.min()) / (img_s.max() - img_s.min())
        img_att_s = show_cam_on_image(img_s, image_relevance_s)
        img_att_s = np.uint8(255 * img_att_s)
        img_att_s = cv2.cvtColor(np.array(img_att_s), cv2.COLOR_RGB2BGR)
        img_att[s, :, :] = img_att_s

    return img_att


def product_of_list_elements(lst):
    """Calculate the product of all elements in a list."""
    product = 1
    for element in lst:
        product *= element
    return product


def attention_to_hex_with_transparency(attention_weight, alpha=0.5):
    # Use the "jet" colormap
    colormap = plt.get_cmap('jet')
    # Map the attention weight to an RGB color
    rgb_color = colormap(attention_weight)[:3]  # Get RGB
    # Convert RGB to hex
    hex_color = mcolors.rgb2hex(rgb_color)
    # Convert alpha to hex and append to hex color
    alpha_hex = format(int(alpha * 255), '02x')
    return f'{hex_color}{alpha_hex}'


def Visualize(model, data_dict, args):

    task_token = 1
    register_token = 3
    img_cls_token = 1
    txt_token = 160

    patch_size = data_dict['patch_size']
    img_size = data_dict['data'].shape[2:]
    img_token_shape = [img_size[0]//patch_size[0], img_size[1]//patch_size[1], img_size[2]//patch_size[2]]

    img_token = product_of_list_elements(img_token_shape)
    require_attn_grad = False

    task_layer_id = 3

    model.eval()
    model.zero_grad()
    img = data_dict['data']

    data_dict['attn_hook'] = True
    data_dict['require_attn_grad'] = require_attn_grad

    attn_map, att_map_txt, output = compute_attention_map(model, data_dict, task_layer_id, task_token,
                                                                      register_token, img_cls_token, img_token, txt_token,
                                                                      img_token_shape, img_size)

    txt_tokens = embed2tokens(list(data_dict['txt_ids'][0].cpu().squeeze().numpy()))[:-1]
    att_map_txt_b = att_map_txt[0]
    txt_mask_b = data_dict['txt_mask'][0].cpu().numpy()
    att_map_txt_b = att_map_txt_b[txt_mask_b > 0]
    att_map_txt_b = att_map_txt_b[1:-2]
    assert len(txt_tokens) == len(att_map_txt_b)
    att_map_txt_b = (att_map_txt_b - att_map_txt_b.min()) / (att_map_txt_b.max() - att_map_txt_b.min() + 1e-32)

    # Generate HTML
    html = "<p style='font-family:Helvetica; font-size:16px;'>"
    for token, attw in zip(txt_tokens, att_map_txt_b):
        token = token.replace('Ġ', ' ')
        hex_color_with_transparency = attention_to_hex_with_transparency(attw,
                                                                         alpha=0.6)  # Adjust alpha as needed
        html += f"<span style='background-color: {hex_color_with_transparency}; margin: 0 2px; padding: 0 2px;'>{token}</span>"
    html += "</p>"

    if 'data_size' in data_dict.keys():
        img_show_size = data_dict['data_size'].squeeze()
    else:
        img_show_size = img.shape

    if len(img_show_size.shape) == 2:
        img_show_size = img_show_size[0]

    img_b = img[0].squeeze().numpy()
    img_b = img_b[:img_show_size[0], :img_show_size[1], :img_show_size[2]]
    if len(attn_map.shape) == 4:
        attn_map_b = attn_map[0, :img_show_size[0], : img_show_size[1], :img_show_size[2]]
    else:
        attn_map_b = attn_map[:img_show_size[0], : img_show_size[1], :img_show_size[2]]

    img_b_att = get_attention_map(img_b, attn_map_b, args)

    return output, html, img_b, img_b_att
