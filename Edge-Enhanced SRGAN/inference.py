import argparse
import os

import PIL.ImageShow
import numpy as np
import sys
import matplotlib.pyplot as plt
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


root_path = r'C:/Users/hlia981/mmdetection/Edge-Enhanced SRGAN/'
image_path = r"C:/Users/hlia981/Downloads/Linnaeus 5 256X256/Linnaeus 5 256X256/test/other"

def bgr_to_rgb(image):
    # Swap the channels using PyTorch indexing
    image_rgb = image[[0, 1, 2], :, :]
    return image_rgb


def lrTransform(image):
    lr_transform = transforms.Compose(
        [
            transforms.Resize((16, 16), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    img_lr = torch.unsqueeze(lr_transform(image), dim=0)
    return img_lr


def load_specific_size_image(path, number=2):
    image_list = []
    for filename in glob.glob(path + '/*.jpg'):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)
        if len(image_list) >= number:
            break
    return image_list


if __name__ == '__main__':
    img_list = load_specific_size_image(
        path=image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(root_path+"sharpen_data", exist_ok=True)
    generator = GeneratorRRDB(3, filters=64, num_res_blocks=23)
    # discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)
    # if torch.cuda.is_available():
    #     generator = generator.cuda()

    generator.load_state_dict(torch.load(root_path + "new_saved_models/generator_88.pth"))
    print("loading model weight")
    for i, img in enumerate(img_list):
        trans = transforms.ToPILImage()
        img_lr = lrTransform(img)
        img_lr = Variable(img_lr.type(torch.FloatTensor))

        gen_hr = generator(img_lr)
        # output, _ = discriminator(gen_hr)
        # print(output.shape)
        # print(output.size)
        img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        img_lr = make_grid(img_lr, nrow=1, normalize=True)
        img_grid = torch.cat((img_lr, gen_hr), -1)
        # img_grid = trans(img_grid)
        # box = yolov8.predict_and_show(img_grid)
        # x, y = box[0], box[1]
        fig, ax = plt.subplots(1, 2)
        class_name = ["bird","dog","cat"]
        ax[0].imshow(trans(gen_hr))
        ax[0].axis('off')  # Hide axes
        ax[0].title.set_text(class_name[1])
        ax[1].imshow(trans(img_lr))
        ax[1].axis('off')  # Hide axes
        ax[1].title.set_text(class_name[2])
        plt.show()
        # img_grid = trans(img_grid)
        # img_grid.show()
        # save_image(img_grid, root_path + f"sharpen_data/new_gen{i}.png", normalize=True)
