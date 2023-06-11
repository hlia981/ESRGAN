import numpy as np
from PIL import Image, ImageFilter
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from torch.nn.functional import conv2d

transform_im = T.ToPILImage()
transform_ts = T.ToTensor()


def sharpen_image_with_unsharp_mask_PIL(image, kernel_size=1, amount=1.0, threshold=1, type=2):
    image = transform_im(image.squeeze(0))
    if type == 1:
        blurred_image = image.filter(ImageFilter.BoxBlur(radius=kernel_size))
    else:
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size))
    blurred_image = np.array(blurred_image)
    image = np.array(image)
    sharpened_image = float(amount + 1) * image - float(amount) * blurred_image
    sharpened_image = np.maximum(sharpened_image, np.zeros(sharpened_image.shape))
    sharpened_image = np.minimum(sharpened_image, 255 * np.ones(sharpened_image.shape))
    sharpened_image = sharpened_image.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred_image) < threshold
        np.copyto(sharpened_image, image, where=low_contrast_mask)

    sharpened_image = transform_ts(sharpened_image)
    return sharpened_image.unsqueeze(0)

def sharpen_image_with_unsharp_mask_opti(image, kernel_size=1, amount=1.0, threshold=1, mask_type=2):
    image = transform_im(image.squeeze(0))
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size)) if mask_type != 1 else image.filter(
        ImageFilter.BoxBlur(radius=kernel_size))
    blurred_image = np.array(blurred_image)
    image = np.array(image)
    sharpened_image = (amount + 1) * image - amount * blurred_image
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred_image) < threshold
        sharpened_image = np.where(low_contrast_mask, image, sharpened_image)
    sharpened_image = transform_ts(sharpened_image)
    return sharpened_image.unsqueeze(0)



def sharpen_image_with_unsharp_mask_PyTorch(image, kernel_size=1, amount=1.0, threshold=1, blur_type=1):
    if blur_type == 1:
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    else:
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256

        kernel = kernel.view(1, 1, kernel_size, kernel_size)

    blurred_image = conv2d(image, kernel, padding=kernel_size // 2)
    sharpened_image = (amount + 1) * image - amount * blurred_image
    sharpened_image = torch.clamp(sharpened_image, min=0, max=1)

    if threshold > 0:
        low_contrast_mask = torch.abs(image - blurred_image) < threshold
        sharpened_image.masked_fill_(low_contrast_mask, image.masked_select(low_contrast_mask))

    sharpened_image = (sharpened_image * 255).round().to(torch.uint8)

    # Rearrange dimensions to [batch, r, g, b]
    # sharpened_image = sharpened_image.permute(0, 2, 3, 1)

    return sharpened_image


def test():
    fig, ax = subplots(1, 4)

    path = r"C:\Users\hlia981\Downloads\Linnaeus 5 256X256\Linnaeus 5 256X256\test\dog\180_256.jpg"
    # image_raw = cv2.imread(path)
    # image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
    # ax[0].imshow(image_raw)
    # ax[0].axis('off')  # Hide axes
    # image = sharpen_image_with_unsharp_mask(image_raw, threshold=1)
    # print(image)
    # # image.show()
    # ax[1].imshow(image)
    # ax[1].axis('off')  # Hide axes

    im1 = Image.open(path)

    im2 = sharpen_image_with_unsharp_mask_PIL(im1)
    ax[2].imshow(im2)
    ax[2].axis('off')

    im3 = sharpen_image_with_unsharp_mask_PIL(im1, type=2)
    ax[3].imshow(im3)
    ax[3].axis('off')

    plt.show()
