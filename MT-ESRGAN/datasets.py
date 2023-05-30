import glob
import random
import os
import numpy as np
from os.path import basename, dirname
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*/*.*"))  # modified to get files within subdirectories

        # create a dictionary to map class names to integers
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(os.listdir(root))}

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path)
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img)

        # extract class name from the file path and encode it
        class_name = basename(dirname(img_path))
        label = self.class_to_index[class_name]

        return {"lr": img_lr, "hr": img_hr, "label": label}

    def __len__(self):
        return len(self.files)
