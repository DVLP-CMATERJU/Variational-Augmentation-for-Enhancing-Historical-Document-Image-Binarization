import os
from PIL import Image
import numpy as np

def splitter(src_path, bn_path, val_split=0.1):

    total_img = os.listdir(src_path)
    valsize = int(val_split * len(total_img))
    save_train = "dsynth_/train/"
    save_val = "dsynth_/val/"
    i = 0
    for color,binary in zip(sorted(os.listdir(src_path)),sorted(os.listdir(bn_path))):
        cname = color.split(".")[0]
        bname = binary.split(".")[0]
        cl_img = np.array(Image.open(src_path+"/"+color))
        bn_img = np.array(Image.open(bn_path+"/"+binary))
        print(color, binary)
        print(i)    
        random_indices = sorted(np.random.choice(total_img, size=valsize, replace=False))

        cl_img = Image.fromarray(cl_img, "RGB")
        bn_img = Image.fromarray(bn_img, "L")
        if i in list(random_indices):
            cl_img.save(save_val+"cl_patches/"+color)
            bn_img.save(save_val+"gt_patches/"+binary)
        else:
            cl_img.save(save_train+f"cl_patches/{cname}.jpg")
            bn_img.save(save_train+f"gt_patches/{bname}.jpg")
        i+=1   
 
splitter("dsynth_/synth_patches","dsynth_/synth_gt")