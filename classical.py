import os 
import numpy as np 
from PIL import Image
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

def gt_otsu(bw_img):
    threshold = threshold_otsu(bw_img)
    bw_img[bw_img>threshold] = 255
    bw_img[bw_img<=threshold] = 0
    return bw_img

def gt_niblack(bw_img):
    threshold = threshold_niblack(bw_img, window_size=127, k=0.2)
    bw_img[bw_img>threshold] = 255
    bw_img[bw_img<=threshold] = 0
    return bw_img

def gt_sauvola(bw_img):
    threshold = threshold_sauvola(bw_img, window_size=127, k=0.2)
    bw_img[bw_img>threshold] = 255
    bw_img[bw_img<=threshold] = 0
    return bw_img

YEAR = 2013
PATH = f"../../dibco_test/{YEAR}/bw_patches/"
SAVE_PATH_OTSU = f"../../dibco_test/{YEAR}/otsu_results/"
SAVE_PATH_NIBL = f"../../dibco_test/{YEAR}/niblack_results/"
SAVE_PATH_SAUV = f"../../dibco_test/{YEAR}/sauv_results/"

for files in sorted(os.listdir(PATH)):
    name = files.split(".")[0]
    img = Image.open(PATH+files)
    img = np.expand_dims(img, axis=-1)
    print(img.shape)
    #for otsu

    bw_img_otsu = gt_otsu(img)
    bw_img_otsu = np.squeeze(bw_img_otsu, axis=-1)
    bw_img_otsu = Image.fromarray(bw_img_otsu, 'L')
    bw_img_otsu.save(f"{SAVE_PATH_OTSU}{name}.jpg")

    #for niblack
    bw_img_nibl = gt_niblack(img)
    bw_img_nibl = np.squeeze(bw_img_nibl, axis=-1)
    bw_img_nibl = Image.fromarray(bw_img_nibl, 'L')
    bw_img_nibl.save(f"{SAVE_PATH_NIBL}{name}.jpg")

    #for sauvola
    bw_img_sauv = gt_sauvola(img)
    bw_img_sauv = np.squeeze(bw_img_sauv, axis=-1)
    bw_img_sauv = Image.fromarray(bw_img_sauv, 'L')
    bw_img_sauv.save(f"{SAVE_PATH_SAUV}{name}.jpg")



