import torch
from util import ConfigFile
import models.m3fm as m3fm
from visualize import Visualize
from data import get_data


def Inference(input_data, vis=False):
    args = ConfigFile(input_data['config_file'])

    torch.backends.cudnn.benchmark = True
    # create model
    args.modalities = list(args.modalities.split(","))
    model = m3fm.__dict__[args.model](**vars(args))
    model.cuda(args.gpu)

    state_dict = torch.load(args.model_weight, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    data_dict = get_data(input_data, args)
    data_dict['gpu'] = args.gpu

    if vis:
        output_dict, txt_html, img, img_att = Visualize(model, data_dict, args)

        for k, v in output_dict.items():
            output_dict[k] = list(v.cpu().squeeze().numpy())

        return output_dict, txt_html, img, img_att
    else:
        model.eval()
        with torch.no_grad():
            output_dict = model(data_dict)

        for k, v in output_dict.items():
            output_dict[k] = list(v.cpu().squeeze().numpy())

        return output_dict
