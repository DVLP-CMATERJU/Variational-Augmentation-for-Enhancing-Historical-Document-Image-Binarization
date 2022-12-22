import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image
import config
from model import build_res_unet

PATH = "../weights/res_unet_bicyc3.pth.tar"
YEAR = 2014
IMG_PATH = f"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_test/{YEAR}/cl_patches/"

SAVE_PATH = f"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_test/{YEAR}/results/"
os.makedirs(SAVE_PATH, exist_ok=True)

gen = build_res_unet(config.DEVICE, n_input=3, n_output=1, size=256)
#gen = nn.DataParallel(gen)
checkpoint = torch.load(PATH, map_location=config.DEVICE)
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

x = torch.rand((1,3,256,256))
y = gen(x).shape
print(y)

for i, filename in enumerate(sorted(os.listdir(IMG_PATH))):
    Img = Image.open(IMG_PATH+filename)
    #convert to tensor
    name = filename.split(".")[0]
    img = config.test_transforms(Img)
    img = img.unsqueeze(0).cuda()

    binarized = (gen(img)).detach()
    binarized = binarized * 0.5 + 0.5

    save_image(binarized, f"{SAVE_PATH}/{name}.jpg")
