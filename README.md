# Variational-Augmentation-for-Enhancing-Historical-Document-Image-Binarization
Official Code Implementation of **[Variational Augmentation of Enhancing Historical Document Image Binarization](https://arxiv.org/abs/2211.06581)** <br>
Accepted at: **[ICVGIP 2022](https://events.iitgn.ac.in/2022/icvgip/accepted_papers.html)** <br>
## Prerequisites
1. Python 3.7+
2. Pytorch 1.9+
3. Albumentations
4. Fast AI

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
- - - - results

- Restoration
- - code
- - - - all relavant files here (this repo)
- - weights
- - - - pretrained/saved weights here
```

## Train Instructions
1. The Augmentation Network (Aug-Net) is based on BicycleGAN. Train the model according to the instructions specified in their official [repository](https://github.com/junyanz/BicycleGAN) using the patches extracted from the training data. Copy the ```checkpoints``` folder into ```synthetic/```.

2. Create a subdirectory ```evaluation/``` to store intermediate results while the model is training.

3. Run ```train.py``` to train the Binarization Network (Bin-Net).

## Inference
1. Change path to the directory containing the test images.
2. Specify path to weight files.
3. Run ```infer.py```.
4. For evaluation, specify the paths to the outputs and the ground truth images in ```eval.py``` and run it. 

## Citation
If you find our paper or code useful, consider citing us:
```
@misc{https://doi.org/10.48550/arxiv.2211.06581,
  doi = {10.48550/ARXIV.2211.06581},
  
  url = {https://arxiv.org/abs/2211.06581},
  
  author = {Dey, Avirup and Das, Nibaran and Nasipuri, Mita},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.6},
  
  title = {Variational Augmentation for Enhancing Historical Document Image Binarization}
```
## Acknowledgements
Our work is partly based on BicycleGAN and we made extensive use of their code. We would like to thank the authors for their contribution.
## TO - DO
- [X] Inference instructions
- [ ] Add environment.yml
- [ ] Add weight files
- [ ] Add sample images
