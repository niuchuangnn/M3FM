# model configs
modalities = 'img,txt,task'
model = 'm3fm_large'
multi_task_head = "v2"
embed_dim = 1024

head_dim_dict = {
    'heart': 1024,
}

task_dim_dict = {
    'heart': 2,
}

num_head_dict = {
    'heart': 1,
}

task_head_map_dict = {
    'heart': 'heart',
}

loss_weight_dict = {
    'heart': [1.0, 1.0],
}

task_base_dict = {
    'heart': -1.0,
}


hu_range_dict_test = {
    'nodule': {'min': [-1300, -1299], 'max': [150, 151]},
    'heart': {'min': [-135, -134], 'max': [225, 226]},
}

gpu = 0
crop_size = [128, 192, 224]
cube_size = [16, 16, 16]
hu_range = [-135, 225]
data_name = ['heart']
task_id = 12
model_weight = 'demo_data/pretrained_models/model_heart.pth'
