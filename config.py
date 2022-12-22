import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_/train"
#"/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/new_synth/bicyc/train/images"
VAL_DIR = "/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/dibco_/val"

LEARNING_RATE = 2e-4
BATCH_SIZE = 12
NUM_WORKERS = 4
L1_LAMBDA = 100
NUM_EPOCHS = 15
PREPOCHS = 5
LOAD_GMODEL = True
LOAD_DMODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "../weights/res_unet_ablnosim.pth.tar"
CHECKPOINT_DISC = "../weights/disc_ablnosim.pth.tar"

test_transforms = transforms.Compose([transforms.Resize(256),  
                                       #transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) # 3 for 3 channel

# FOR training dibco
dibco_transforms = A.Compose(
    [
        A.OneOf([A.Resize(width=256, height=256, p=0.5), 
                A.RandomCrop(256, 256, p=0.5)], p=1.0),
        A.HorizontalFlip(p=0.5)
     ],
    additional_targets={"image0": "image"},
)
'''
# For training synthetic/mixed
syn_transforms = A.Compose(
    [
        A.Resize(width=256, height=256, p=1.0),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5)
    ], additional_targets={"image0": "image"},
)
'''
transform_only_output = A.Compose(
    [
        #A.Resize(width=256, height=256, p=1.0),
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_input = A.Compose(
    [
        #A.Resize(width=256, height=256, p=1.0),
        A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5,], std=[0.5,], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)