import cv2
import numpy as np
from PIL import Image, ImageFilter
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

def sharpen_image_with_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened_image = float(amount + 1) * image - float(amount) * blurred_image
    sharpened_image = np.maximum(sharpened_image, np.zeros(sharpened_image.shape))
    sharpened_image = np.minimum(sharpened_image, 255 * np.ones(sharpened_image.shape))
    sharpened_image = sharpened_image.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred_image) < threshold
        np.copyto(sharpened_image, image, where=low_contrast_mask)
    return Image.fromarray(sharpened_image)

def sharpen_image_with_unsharp_mask_PIL(image, kernel_size=1, amount=1.0, threshold=1, type=1):
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
    return Image.fromarray(sharpened_image)

def test():
    fig, ax = subplots(1, 4)

    path = r"C:\Users\hlia981\Downloads\Linnaeus 5 256X256\Linnaeus 5 256X256\test\dog\180_256.jpg"
    image_raw = cv2.imread(path)
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
    ax[0].imshow(image_raw)
    ax[0].axis('off')  # Hide axes
    image = sharpen_image_with_unsharp_mask(image_raw, threshold=1)
    print(image)
    # image.show()
    ax[1].imshow(image)
    ax[1].axis('off')  # Hide axes

    im1 = Image.open(path)

    im2 = sharpen_image_with_unsharp_mask_PIL(im1)
    ax[2].imshow(im2)
    ax[2].axis('off')

    im3 = sharpen_image_with_unsharp_mask_PIL(im1, type=2)
    ax[3].imshow(im3)
    ax[3].axis('off')

    plt.show()