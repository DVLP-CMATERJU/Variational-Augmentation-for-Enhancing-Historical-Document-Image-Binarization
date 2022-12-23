# Variational-Augmentation-for-Enhancing-Historical-Document-Image-Binarization
Official Code Implementation of **Variational Augmentation of Enhancing Historical Document Image Binarization** <br>
Submitted to: **ICVGIP 2022** <br>
## Prerequisites

## Dataset Download
1. You can download the training images of DIBCO from [here](https://drive.google.com/file/d/1tpgxnHPHpwA9F39WNauucwtCDcLV7fRx/view?usp=share_link). Extract patches using ```datamaker.py```.
2. You can download the testing data from [here](https://drive.google.com/file/d/1tpgxnHPHpwA9F39WNauucwtCDcLV7fRx/view?usp=share_link).
3. You can also download the training patches directly from [here](https://drive.google.com/file/d/1tpgxnHPHpwA9F39WNauucwtCDcLV7fRx/view?usp=share_link). (recommended)
## Directory Structure
```
- training_datasets
- - train
- - - - bw_patches
- - - - gt_patches
- - - - cl_patches
- - val
- - - - bw_patches
- - - - gt_patches
- - - - cl_patches

- testing_datasets
- - <DIBCO_YEAR>
- - - - bw_patches
- - - - gt_patches
- - - - cl_patches

- Restoration
- - code
- - - - all relavant files here (this repo)
- - weights
- - - - pretrained/saved weights here
```

## Train Instructions
1. The Augmentation Network (Aug-Net) is based on BicycleGAN. Train the model according to the instructions specified in their official repository using the patches extracted from the training data. Copy the ```checkpoints``` folder into ```synthetic/```.

2. Create an= subdirectory ```evaluation/``` to store intermediate results while the model is training.

3. Run ```train.py``` to train the Binarization Network (Bin-Net).

## Inference
