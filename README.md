# M3FM

<div style="text-align: justify"> M3FM (medical multimodal-multitask foundation model) is the first-of-its-kind foundation model architecture with an application in lung cancer screening (LCS).
M3FM can flexibly adapt to new tasks with a small out-of-distribution dataset, effectively handle various combinations of multimodal data, and efficiently process high-dimensional images at multiple scales.
In a broader sense, as a specialty-oriented generalist medical AI (SOGMAI) model, M3FM innovates lung cancer management and related tasks.
This SOGMAI approach paves the way for similar breakthroughs in other areas of medicine, closing the gap between specialists and the generalist. </div>



## Installation
Assuming [Anaconda](https://www.anaconda.com/) with Python 3.9 installed, the following are step-by-step commands for setting up the environment:

```shell script
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.17.0
pip install fairscale
pip install opencv-python
pip install matplotlib==3.7.1

git clone https://github.com/niuchuangnn/M3FM.git
cd M3FM
```

## Prepare Data
Download the zip file through this [link](https://drive.google.com/uc?export=download&id=1QJer00vxumElsZIvdpdJLcD_6jeCbX4j), manually unzip or run ```python unzip_demo_data.py``` to unzip it under the root folder. Then, you will have ```~/M3FM/demo_data```.

## Run Demo

The first example is the prediction of lung cancer risk with multimodal data from the NLST dataset. The ground truth is defined as the clinical evidence that the patient was diagnosed with lung cancer within 1 year after screening.
Here are the inputs:

```shell
input_data = {
    'ct_path': 'demo_data/ct_npy/cancer_risk.npy',  # CT data
    'pixel_size': [2.0, 0.53, 0.53],  # physical size of CT voxels
    'coords': {'cancer_risk': [7, 132, 112, 450, 233, 490]},  # lung bounding box coordinates
    'clinical_txt': "The patient is 68.0 years old. Gender is Female. Race is white. Ethnicity is neither hispanic nor latino. Height is 62.0 inches. Weight is 134.0 pounds. Education is associate degree/ some college. Current smoker. Smoking duration is 37.5 pack years. Smoking intensity is 15.0 cigarettes per day. The patient had diabetes diagnosed at 68.0 years old. The patient had hypertension diagnosed at 64.0 years old. Patient's brother(s) (including half-brothers) have lung cancer. Patient's mother have lung cancer.",
    'question': 'Predict the lung cancer risk over six years.',
    'config_file': 'config_files/config_m3mf_cancer_risk.py'
}
```
The second example is the prediction of cardiovascular abnormality with multimodal data from the NLST dataset. The patient was reported having cardiovascular abnormality.
```shell script
input_data = {
    'ct_path': 'demo_data/ct_npy/heart.npy', # CT data
    'pixel_size': [2.50, 0.57, 0.57], # physical size of CT voxels
    'coords': {'heart': [52, 119, 120, 290, 189, 428]},
    'clinical_txt': "The patient is 64.0 years old. Gender is Female. Race is white. Ethnicity is neither hispanic nor latino. Height is 65.0 inches. Weight is 130.0 pounds. Education is high school graduate/ged. Former smoker. Smoking duration is 46.0 pack years. Smoking intensity is 20.0 cigarettes per day. 1.0 years since quit smoking. The patient had hypertension diagnosed at 56.0 years old. The patient had pneumonia diagnosed at 61.0 years old. The patient had stroke diagnosed at 60.0 years old. Patient's mother have lung cancer.",
    'question': 'Is there any significant cardiovascular abnormality?',
    'config_file': 'config_files/config_m3mf_heart.py'
}
```

Use the following APIs to get the predictions:

```shell
from inference import Inference
output_dict = Inference(input_data)

## lung cancer risk prediction
print('Output', output_dict)
# {'cancer_risk': [0.9028023, 0.75654167, 0.70547664, 0.5618707, 0.52376777, 0.5138244]}
# the cancer risk over 6 years.

## Cardiovascular abnormality
print('Output', output_dict['heart'][1])
# CVD score: 0.999666
```

Use the following APIs to obtain both predictions and relevance maps on CT and clinical text:

```shell
from inference import Inference
output_dict, txt_html, img, img_att = Inference(input_data, vis=True)
```
Here ```txt_html``` is the clinical text with heatmap colors the in HTML format, ```img``` is the input CT data, and ```img_att``` is the input CT data with heatmap colors.

Check [inference_demo_note.ipynb](https://github.com/niuchuangnn/M3FM/blob/main/inference_demo_note.ipynb) for visualization with both CT and clinical text.

## License
This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation

```shell
@misc{niu2024medical,
      title={Medical Multimodal-Multitask Foundation Model for Lung Cancer Screening}, 
      author={Chuang Niu and Qing Lyu and Christopher D. Carothers and Parisa Kaviani and Josh Tan and Pingkun Yan and Mannudeep K. Kalra and Christopher T. Whitlow and Ge Wang},
      year={2024},
      eprint={2304.02649},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
