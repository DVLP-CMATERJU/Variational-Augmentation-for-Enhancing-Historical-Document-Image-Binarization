from math import sqrt, log10
import numpy as np
import math
import cv2
from bwmorph_thin import bwmorph_thin as bwmorph

'''
DRD adapted from Peb Ruswono Aryan's DIBCO 2016 metrics


def get_drd(img, gt):

	#img, gt = img[:,:,0], gt[:,:,0]
	height, width = img.shape
	neg = np.zeros(img.shape)
	neg[gt!=img] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), img.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.0
	W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	W[n,n] = 1.0
	W = 1.0/W
	W[n,n] = 0.0
	W /= W.sum()

	nubn = 1e-8
	block_size = 8
	for y1 in range(0, height, block_size):
		for x1 in range(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1 - gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.0
	tmp = np.zeros(W.shape)
	for i in range(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(img[y[i],x[i]] - gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn

def get_psnr(img, gt):

    mse = np.mean((img - gt)**2)
    if mse == 0:
        return 100
    maxp = 255.0
    psnr = 10 * log10(maxp / sqrt(mse))

    return psnr
'''
def my_xor_infile(u_infile, u0_GT_infile):
    temp_fp_infile = np.zeros(u_infile.shape, np.uint8)
    temp_fp_infile[(u_infile == 0) & (u0_GT_infile == 1)] = 1

    temp_fn_infile = np.zeros(u_infile.shape, np.uint8)
    temp_fn_infile[(u_infile == 1) & (u0_GT_infile == 0)] = 1

    temp_xor_infile = (temp_fp_infile | temp_fn_infile)
    return temp_xor_infile


def get_drd(im, im_gt):
    xm, ym = im.shape
    
    # get NUBN
    blkSize=8 # even number
    # 1, 1 padding 후 중앙
    u0_GT1 = np.zeros((xm + 2, ym + 2), np.uint8)
    u0_GT1[1 : xm + 1, 1 : ym + 1] = im_gt
    NUBN = 0
    blkSizeSQR = blkSize * blkSize
    # matlab 은 배열 인덱스 1 부터 시작, for 문 마지막 포함
    for i in range(1, (xm - blkSize + 2), blkSize):
        for j in range(1, (ym - blkSize + 2), blkSize):
            blkSum = np.sum(u0_GT1[i:i+blkSize, j:j+blkSize])
            if blkSum != 0 and blkSum != blkSizeSQR:
                NUBN += 1

    mask_size = 5 # odd number
    wm = np.zeros((mask_size, mask_size))
    ic = int(mask_size / 2) # center coordinate
    jc = ic
    for i in range(mask_size):
        for j in range(mask_size):
            if i == ic and j == jc:
                continue
            wm[i, j] = 1. / math.sqrt( (i - ic) * (i - ic) + (j - jc) * (j - jc) )
    wm[ic, jc] = 0.
    wnm = wm / np.sum(wm) # % Normalized weight matrix
    
    # 1 ~ xm + 3, 2 ~ xm + 1
    # get sum of DRD_k
    # 2칸씩 padding 후 가운데
    u0_GT_Resized = np.zeros((xm + ic + 2, ym + jc + 2), np.uint8)
    u0_GT_Resized[ic : xm+ic, jc : ym+jc] = im_gt
    u_Resized = np.zeros((xm + ic + 2, ym + jc + 2), np.uint8)
    u_Resized[ic : xm+ic, jc : ym+jc] = im

    temp_fp_Resized = np.zeros(u_Resized.shape, np.uint8)
    temp_fp_Resized[(u_Resized == 0) & (u0_GT_Resized == 1)] = 1
    temp_fn_Resized = np.zeros(u_Resized.shape, np.uint8)
    temp_fn_Resized[(u_Resized == 1) & (u0_GT_Resized == 0)] = 1

    Diff = temp_fp_Resized | temp_fn_Resized
    xm2, ym2 = Diff.shape
    SumDRDk = 0.
    for i in range(ic, xm2 - ic):
        for j in range(jc, ym2 - jc):
            if Diff[i, j] == 1:
                Local_Diff = my_xor_infile(u0_GT_Resized[i - ic : i + ic + 1 , j - ic : j + ic + 1], u_Resized[i:i+1, j:j+1])
                DRDk = np.sum(np.multiply(Local_Diff, wnm))
                SumDRDk += DRDk

    temp_DRD = SumDRDk / NUBN

    return temp_DRD
def get_psnr(img, gt):

    # true positive
    tp = np.zeros(gt.shape, np.uint8)
    tp[(img==0) & (gt==0)] = 1
    numtp = tp.sum()

    # false positive
    fp = np.zeros(gt.shape, np.uint8)
    fp[(img==0) & (gt==1)] = 1
    numfp = fp.sum()

    # false negative
    fn = np.zeros(gt.shape, np.uint8)
    fn[(img==1) & (gt==0)] = 1
    numfn = fn.sum()

    h, w = gt.shape
    npixel = h * w
    mse = float(numfp + numfn) / npixel
    psnr = 10. * np.log10(1. / mse)
    return psnr

def get_fmeasure(img, gt):

    # true positive
    tp = np.zeros(gt.shape, np.uint8)
    tp[(img==0) & (gt==0)] = 1
    numtp = tp.sum()

    # false positive
    fp = np.zeros(gt.shape, np.uint8)
    fp[(img==0) & (gt==1)] = 1
    numfp = fp.sum()

    # false negative
    fn = np.zeros(gt.shape, np.uint8)
    fn[(img==1) & (gt==0)] = 1
    numfn = fn.sum()
    precision = (numtp + 1e-7) / float(numtp + numfp + 1e-7)
    recall = (numtp + 1e-7) / float(numtp + numfn + 1e-7)
    fmeasure = 100. * (2. * recall * precision) / (recall + precision)

    return fmeasure

def get_fpmeasure(img, gt):

    # get skeletonized im_gt
    sk = bwmorph(1 - gt)
    im_sk = np.ones(gt.shape, np.uint8)
    im_sk[sk] = 0

    # true positive
    tp = np.zeros(gt.shape, np.uint8)
    tp[(img==0) & (gt==0)] = 1
    numtp = tp.sum()

    # false positive
    fp = np.zeros(gt.shape, np.uint8)
    fp[(img==0) & (gt==1)] = 1
    numfp = fp.sum()

    # skel true positive
    ptp = np.zeros(gt.shape, np.uint8)
    ptp[(img==0) & (im_sk==0)] = 1
    numptp = ptp.sum()

    # skel false negative
    pfn = np.zeros(gt.shape, np.uint8)
    pfn[(img==1) & (im_sk==0)] = 1
    numpfn = pfn.sum()

    # get pseudo-FMeasure
    precision = (numtp + 1e-7) / float(numtp + numfp + 1e-7)
    precall = (numptp + 1e-7) / float(numptp + numpfn + 1e-7)
    fpmeasure = 100 * (2 * precall * precision) / (precall + precision) # percent

    return fpmeasure

