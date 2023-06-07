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


root_path = r"C:/Users/GGPC/PycharmProjects/pythonProject/pyTorchLearning/ESRGAN/"
image_path = r"D:\google_downloads\Linnaeus 5 256X256\test\bird"

def bgr_to_rgb(image):
    # Swap the channels using PyTorch indexing
    image_rgb = image[[0, 1, 2], :, :]
    return image_rgb


def lrTransform(image):
    lr_transform = transforms.Compose(
        [
            transforms.Resize((256//4, 256//4), transforms.InterpolationMode.BICUBIC),
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

def upsample_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorRRDB(3, filters=64, num_res_blocks=23).to(device)
    # discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)
    generator.load_state_dict(torch.load(root_path + "saved_models/generator_33.pth"))
    # discriminator.load_state_dict(torch.load(root_path + "saved_models/discriminator_33.pth"))
    img_lr = lrTransform(image)
    img_lr = Variable(img_lr.type(torch.cuda.FloatTensor))
    gen_hr = generator(img_lr)
    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    trans = transforms.ToPILImage()
    img_grid = trans(gen_hr)
    img_grid.show()

if __name__ == '__main__':
    im = Image.open(r"C:\Users\GGPC\PycharmProjects\pythonProject\pyTorchLearning\Yolov8\section1.png")
    upsample_image(im)
    # img_list = load_specific_size_image(
    #     path=image_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.makedirs(root_path+"sharpen_data", exist_ok=True)
    # generator = GeneratorRRDB(3, filters=64, num_res_blocks=23).to(device)
    # discriminator = Discriminator(input_shape=(3, 256, 256)).to(device)
    # # if torch.cuda.is_available():
    # #     generator = generator.cuda()
    #
    # generator.load_state_dict(torch.load(root_path + "saved_models/generator_33.pth"))
    # discriminator.load_state_dict(torch.load(root_path + "saved_models/discriminator_33.pth"))
    # print("loading model weights")
    # for i, img in enumerate(img_list):
    #     trans = transforms.ToPILImage()
    #     img_lr = lrTransform(img)
    #     img_lr = Variable(img_lr.type(torch.cuda.FloatTensor))
    #
    #     gen_hr = generator(img_lr)
    #     _, classification = discriminator(gen_hr)
    #     output = torch.argmax(classification, dim=1)
    #     img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
    #     gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
    #     img_lr = make_grid(img_lr, nrow=1, normalize=True)
    #     img_grid = torch.cat((img_lr, gen_hr), -1)
    #     # img_grid = trans(img_grid)
    #     # box = yolov8.predict_and_show(img_grid)
    #     # x, y = box[0], box[1]
    #     fig, ax = plt.subplots(1, 2)
    #     class_name = ["bird","dog","flower"]
    #     ax[0].imshow(trans(img_lr))
    #     ax[0].axis('off')  # Hide axes
    #     ax[0].title.set_text(class_name[output.item()])
    #     ax[1].imshow(trans(gen_hr))
    #     ax[1].axis('off')  # Hide axes
    #     ax[1].title.set_text(class_name[output.item()])
    #     plt.show()
    #     # img_grid = trans(img_grid)
    #     # img_grid.show()
    #     # save_image(img_grid, root_path + f"sharpen_data/new_gen{i}.png", normalize=True)