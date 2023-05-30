import argparse
import os

import PIL.ImageShow
import numpy as np
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
import matplotlib.pyplot as plt
import glob
import torch
from PIL import Image, ImageDraw


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


root_path = 'C:/Users/hlia981/mmdetection/ESRGAN/'


def bgr_to_rgb(image):
    # Swap the channels using PyTorch indexing
    image_rgb = image[[0, 1, 2], :, :]
    return image_rgb


def lrTransform(image):
    lr_transform = transforms.Compose(
        [
            transforms.Resize((256 // 4, 256 // 4), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    img_lr = torch.unsqueeze(lr_transform(image), dim=0)
    return img_lr


def load_all_image(path):
    image_list = []
    for filename in glob.glob(path + '/*.jpg'):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)
    return image_list


if __name__ == '__main__':
    img_list = load_all_image(
        path="C:/Users/hlia981/Downloads/Linnaeus 5 256X256/Linnaeus 5 256X256/train/small_dataset")
    i = 0
    os.makedirs(root_path+"data", exist_ok=True)
    for img in img_list:
        i += 1
        trans = transforms.ToPILImage()
        # img = Image.open(root_path+'data/121_256.jpg')

        img_lr = lrTransform(img)
        img_lr = Variable(img_lr.type(torch.cuda.FloatTensor))

        generator = GeneratorRRDB(3, filters=64, num_res_blocks=23)
        if torch.cuda.is_available():
            generator = generator.cuda()

        generator.load_state_dict(torch.load(root_path + "saved_models/generator_53.pth"))
        print("loading model weight")
        gen_hr = generator(img_lr)
        img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        img_lr = make_grid(img_lr, nrow=1, normalize=True)
        img_grid = torch.cat((img_lr, gen_hr), -1)
        # img_grid = trans(img_grid)
        # box = yolov8.predict_and_show(img_grid)
        # x, y = box[0], box[1]
        # img_grid = trans(img_grid)
        # img_grid.show()
        save_image(img_grid, root_path + f"data/new_gen{i}.png", normalize=True)

    #
    # gen_hr = bgr_to_rgb(gen_hr)
    # img_grid = torch.cat((img_lr, gen_hr), -1)
    # # print(img_grid.type(), img_grid.size())
    #
    # img_grid = trans(img_grid)
    # img_grid.show()

    # save_image(img_grid, root_path+"data/%d.png", normalize=False)

    # trans = transforms.ToPILImage()
    # gen_hr = trans(gen_hr)
    # # print(gen_hr)
    # gen_hr.show()

    # B,G,R -> R,G,B
