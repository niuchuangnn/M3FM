from inference import Inference
import matplotlib.pyplot as plt
import numpy as np


def show_image_att(imgs, img_atts):
    for s in range(0, imgs.shape[0], 4):
        img = imgs[s]
        img_att = img_atts[s]
        fig, axs = plt.subplots(1, 2)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.array([img, img, img]).transpose([1, 2, 0])
        axs[0].imshow(img)
        axs[0].axis('off')

        axs[1].imshow(img_att)
        axs[1].axis('off')
    plt.show()


## cancer risk demo
input_data = {
    'ct_path': 'demo_data/ct_npy/cancer_risk.npy',
    'pixel_size': [2.0, 0.53, 0.53],
    'coords': {'cancer_risk': [7, 132, 112, 450, 233, 490]},
    'clinical_txt': "The patient is 68.0 years old. Gender is Female. Race is white. Ethnicity is neither hispanic nor latino. Height is 62.0 inches. Weight is 134.0 pounds. Education is associate degree/ some college. Current smoker. Smoking duration is 37.5 pack years. Smoking intensity is 15.0 cigarettes per day. The patient had diabetes diagnosed at 68.0 years old. The patient had hypertension diagnosed at 64.0 years old. Patient's brother(s) (including half-brothers) have lung cancer. Patient's mother have lung cancer.",
    'question': 'Predict the lung cancer risk over six years.',
    'config_file': 'config_files/config_m3mf_cancer_risk.py'
}

output_dict, txt_html, img, img_att = Inference(input_data, vis=True)
show_image_att(img, img_att)
print(output_dict)


## CVD diagnosis demo
input_data = {
    'ct_path': 'demo_data/ct_npy/heart.npy',
    'pixel_size': [2.50, 0.57, 0.57],
    'coords': {'heart': [52, 119, 120, 290, 189, 428]},
    'clinical_txt': "The patient is 64.0 years old. Gender is Female. Race is white. Ethnicity is neither hispanic nor latino. Height is 65.0 inches. Weight is 130.0 pounds. Education is high school graduate/ged. Former smoker. Smoking duration is 46.0 pack years. Smoking intensity is 20.0 cigarettes per day. 1.0 years since quit smoking. The patient had hypertension diagnosed at 56.0 years old. The patient had pneumonia diagnosed at 61.0 years old. The patient had stroke diagnosed at 60.0 years old. Patient's mother have lung cancer.",
    'question': 'Is there any significant cardiovascular abnormality?',
    'config_file': 'config_files/config_m3mf_heart.py'
}

output_dict, txt_html, img, img_att = Inference(input_data, vis=True)
show_image_att(img, img_att)
print('CVD score:', output_dict['heart'][1])


