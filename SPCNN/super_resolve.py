from __future__ import print_function
import argparse
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop

import numpy as np

input_image = r"C:\Users\hlia981\Downloads\Linnaeus 5 256X256\Linnaeus 5 256X256\bird\test\1083_256.jpg"
model = r'C:\Users\hlia981\mmdetection\SPCNN\model_epoch_30.pth'
output_filename = 'out.png'


# Training settings
# parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--input_image', type=str, required=True, help='input image to use')
# parser.add_argument('--model', type=str, required=True, help='model file to use')
# parser.add_argument('--output_filename', type=str, help='where to save the output image')
# parser.add_argument('--cuda', action='store_true', help='use cuda')
# opt = parser.parse_args()
# 
# print(opt)

def input_transform(input, crop_size=256, upscale_factor=3):
    transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])
    return transform(input)


def super_resolve(input_image):
    if isinstance(input_image, str):
        # If it is a string, we will assume it's a path and attempt to open it
        try:
            img = Image.open(input_image).convert('YCbCr')
            print(f"The input is a path to an image.")
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"The input is a path but not an image.")
        except:
            print(f"Error while trying to open the file.")
    elif isinstance(input_image, Image.Image):
        # If it's a PIL Image object, confirm this
        img = input_image.convert('YCbCr')
        print(f"The input is a PIL Image.")
    else:
        # If it's neither, return an error message
        print(f"The input is neither a path to a file nor a PIL Image.")
        return None

    y, cb, cr = img.split()

    sresolver = torch.load(model)
    img_to_tensor = ToTensor()
    # input = input_transform(y)
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    # if cuda:
    #     model = model.cuda()
    #     input = input.cuda()

    out = sresolver(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    # out_img.show()
    # out_img.save(output_filename)
    # print('output image saved to ', output_filename)
    return out_img
