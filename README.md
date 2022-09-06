# 서울 랜드마크 이미지 분류 경진대회
Dacon Basic | CV | Classification | Landmark | Accuracy<br>
랜드마크 이미지 데이터셋을 이용하여 랜드마크의 라벨을 분류

<br>[Competition Link](https://dacon.io/competitions/official/235957/overview/description)
* 주최/주관: Dacon
* **Private 4th, Score 1.0**
***
## Structure
Train/Test data folder and sample submission file must be placed under **dataset** folder.
```
repo
  |——dataset
        |——train
                |——001.PNG
                |——....
        |——test
                |——001.PNG
                |——....
        |——train.csv
        |——test.csv
        |——sample_submission.csv
  kfold_main.py
  kfold_inference.py
  constant.py
  requirements.txt
```
***
## Development Environment
* Windows 10
* i9-10900X
* RTX 2080Ti 1EA
* CUDA 11.4
***
## Environment Settings

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-385/)

### Requirements
```shell
pip install --upgrade -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
***
## Train
```shell
python kfold_main.py
```
***
## Inference
```shell
python kfold_inference.py
```
***
## Solution
### Training Details
* backbone: resnet50d
* 5 fold(stratified)
* lr: 1e-4
* epochs: 200
* mixup epochs: 180
* batch size: 8
* optimizer: AdamW~
* image size: 224
* scheduler: CosineAnnleaingLR (T_max: 10, eta_min: 1e-6)
* label smoothing: 0.1
***
## Tried Techniques
* Hard Crop Augmentation (224, 256, 320)