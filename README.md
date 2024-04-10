# M3FM (In Updating)
M3FM (medical multimodal-multitask foundation model) is the first-of-its-kind foundation model architecture with an application in lung cancer screening (LCS).

## Installation
Assuming [Anaconda](https://www.anaconda.com/) with python 3.9 installed, a step-by-step to set up environment is as follows:

```shell script
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.17.0
pip install fairscale
pip install opencv-python
pip install matplotlib==3.7.1

git clone https://github.com/niuchuangnn/M3FM.git
cd M3FM
```

## Prepare Demo Data
Download the zip file through this [link](https://drive.google.com/uc?export=download&id=1QJer00vxumElsZIvdpdJLcD_6jeCbX4j), unzip it to the root folder, and then you have ```~/M3FM/demo_data```.

## Run Demo

```shell script
python inference_demo.py
```

or check ```inference_demo_not.ipynb```

### License
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
