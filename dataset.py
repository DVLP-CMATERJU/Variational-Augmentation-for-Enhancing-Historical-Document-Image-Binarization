import numpy as np
import config
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class BinDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.target_dir = target_dir
        self.source_dir = source_dir
        self.transform = transform
        self.list_targets = os.listdir(self.target_dir)
        self.list_sources = os.listdir(self.source_dir)

    def __len__(self):
        return len(self.list_sources)

    def __getitem__(self, index):
        src_file = self.list_sources[index]
        tgt_file = self.list_targets[index]
        src_path = os.path.join(self.source_dir, src_file)
        tgt_path = os.path.join(self.target_dir, tgt_file)
        src_image = np.array(Image.open(src_path))
        tgt_image = np.array(Image.open(tgt_path).convert("L"))

        #augmentations = config.transform(image=src_image, image0=tgt_image)
        if self.transform:
            augmentations = self.transform(image=src_image, image0=tgt_image)
            src_image = augmentations["image"]
            tgt_image = augmentations["image0"]
                
        src_image = config.transform_only_input(image=src_image)["image"]
        tgt_image = config.transform_only_output(image=tgt_image)["image"]

        return src_image, tgt_image

class SynDataset(Dataset):
    def __init__(self, source_dir, transform=None):
        self.source_dir = source_dir
        self.transform = transform
        self.list_sources = os.listdir(self.source_dir)

    def __len__(self):
        return len(self.list_sources)

    def __getitem__(self, index):
        src_file = self.list_sources[index]
        src_path = os.path.join(self.source_dir, src_file)
        src_image = np.array(Image.open(src_path))[:,512:,:]
        tgt_image = np.array(Image.open(src_path).convert("L"))[:,:512]

        #augmentations = config.transform(image=src_image, image0=tgt_image)
        if self.transform:
            augmentations = self.transform(image=src_image, image0=tgt_image)
            src_image = augmentations["image"]
            tgt_image = augmentations["image0"]
                
        src_image = config.transform_only_input(image=src_image)["image"]
        tgt_image = config.transform_only_output(image=tgt_image)["image"]

        return src_image, tgt_image

def test():
    PATH = "/media/Reserve_Storage/student_data/intern/intern_1/data/BINARIZATION/new_synth/bicyc/train/"
    train_dataset = SynDataset(source_dir=PATH, transform=config.syn_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
#test()