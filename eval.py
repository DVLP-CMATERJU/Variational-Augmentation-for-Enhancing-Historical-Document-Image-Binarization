import numpy as np
import torch
import cv2
from PIL import Image
import os
from metrics import *

def evaluate(img_dir, gt_dir):

	fsum, pfsum, psum, dsum = 0, 0, 0, 0
	num = 0
	#a_img = [img for img in os.listdir(img_dir) if "fake" in img]
	#print(len(a_img)) 
	#print(len(os.listdir(gt_dir)))
	
	for IMG, GT in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(gt_dir))):
		img = Image.open(img_dir+IMG).convert('L')
		img = np.array(img)
		#print(IMG)
		#print(IMG)
		gt = Image.open(gt_dir+GT).convert('L')
		gt = np.array(gt)
		if np.mean(gt) < 254: # and IMG != "2_patch8.png" and IMG != "4_patch8.png":
			num +=1
			img[img>128] = 255
			img[img<=128] = 0
			#img = img[:,:,0] #comment for classical algos

			gt[gt>128] = 255
			gt[gt<=128] = 0
			img, gt = (img/255).astype(np.uint8), (gt/255).astype(np.uint8)
			psnr = get_psnr(img, gt)
			fm, pfm = get_fmeasure(img, gt), get_fpmeasure(img, gt)
			drd = get_drd(img, gt)

			fsum += fm
			pfsum += pfm
			psum += psnr
			dsum += drd

	return fsum/num, pfsum/num, psum/num, dsum/num

YEAR = 2019
img_dir = f"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_test/{YEAR}/results/"
#img_dir = "/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/DocumentBinarization/DIBCO/predicted_image_dibco/step2_normal/19/"
#img_dir ="/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/pytorch-CycleGAN-and-pix2pix/results/dibco/test_latest/images/"
#img_dir = f"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_test/{YEAR}/niblack_results/"
gt_dir = f"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_test/{YEAR}/gt_patches/"
scores = evaluate(img_dir, gt_dir)
print("F-Measure: {:.3f}\nFp-Measure: {:.3f}\nPSNR: {:.3f}\nDRD: {:.3f}" \
	.format(scores[0],scores[1],scores[2],scores[3]))