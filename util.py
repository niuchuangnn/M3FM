import importlib.util
import sys
from transformers import RobertaTokenizer
import numpy as np
txt_tokenizer = RobertaTokenizer.from_pretrained('demo_data/pretrained_models/roberta-base')


def import_py_file(path):
    # Create a module name based on the file path
    module_name = path.replace('/', '_').replace('.py', '')

    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, path)

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    return module


def extract_variables_from_module(path):
    module = import_py_file(path)
    variables_dict = {}

    for attribute_name in dir(module):
        if not attribute_name.startswith('__'):
            attribute_value = getattr(module, attribute_name)

            if not callable(attribute_value) and not isinstance(attribute_value, type(sys)):
                variables_dict[attribute_name] = attribute_value

    return variables_dict


def get_default_configs(args, config_file):
    if config_file is not None:
        config_data = extract_variables_from_module(config_file)
        for k, v in config_data.items():
            setattr(args, k, v)
    return args


class ConfigFile:
    def __init__(self, config_file):
        config_data = extract_variables_from_module(config_file)
        for k, v in config_data.items():
            setattr(self, k, v)


def txt2embed(txt_list, max_length=50):
    encoding_txt = txt_tokenizer(
        txt_list,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    return encoding_txt


def embed2tokens(ids):
    tokens = txt_tokenizer.convert_ids_to_tokens(
        ids,
        skip_special_tokens=True,
    )
    return tokens


def tokens2txt(tokens):
    txt = txt_tokenizer.convert_tokens_to_string(tokens)
    return txt


def get_sincos_size_embed(embed_dim, size):
    dim_hw = embed_dim // 6 * 2
    dim_s = (embed_dim - dim_hw * 2)

    emb_s = get_1d_sincos_size_embed(dim_s, size[:, 0])
    emb_h = get_1d_sincos_size_embed(dim_hw, size[:, 1])  # (M, D/2)
    emb_w = get_1d_sincos_size_embed(dim_hw, size[:, 2])  # (M, D/2)

    emb = np.concatenate([emb_s, emb_h, emb_w], axis=1)  # (M, D)
    return emb


def get_1d_sincos_size_embed(embed_dim, size):
    """
    embed_dim: output dimension for each position
    size: a list of sizes to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2).astype(np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    size = size.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', size, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
