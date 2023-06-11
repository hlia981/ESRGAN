import torch
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from another_edge_e import *
from pytorch_sharpen import *
from sharpenTool import *

class EdgeExtractor(nn.Module):
    def __init__(self):
        super(EdgeExtractor, self).__init__()
        # Sobel operator kernels
        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).repeat(3, 1, 1, 1))
        self.conv_y.weight = nn.Parameter(
            torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).repeat(3, 1, 1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        edge_x = torch.abs(self.conv_x(img))
        edge_y = torch.abs(self.conv_y(img))
        edge = edge_x + edge_y
        # Normalize the edges to [0, 1]
        edge = (edge - edge.min()) / (edge.max() - edge.min())
        # Apply the edges back to the original image
        img_with_edge = img * edge
        return img_with_edge


def sharpen_image(image: torch.Tensor, kernel_size: int = 5, sigma: float = 2., alpha: float = 1.5):
    """
    Apply an unsharp mask to a 4D tensor (image).
    image : torch.Tensor
        A 4D tensor that represents a batch of images. The dimensions should be (batch_size, channels, height, width).
    kernel_size : int, optional
        The size of the blur kernel.
    sigma : float, optional
        The standard deviation of the Gaussian function used to create the blur kernel.
    alpha : float, optional
        The amount to scale the mask when adding it to the original image. A higher value will make the sharpening stronger.
    """
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min)
    # Ensure the image is in the right range (0-1).
    if image.min() < 0 or image.max() > 1:
        print(image.min, image_max)

    # assert image.min() >= 0 and image.max() <= 1, 'The image should be in the range [0, 1]'

    # Create the blur kernel.
    gauss = kornia.filters.GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))

    # Use padding to ensure the output image has the same size as the input.
    padding = kernel_size // 2

    # Create the blurred image.
    blurred = gauss(image)

    # Create the mask.
    mask = image - blurred

    # Apply the mask to the original image.
    sharpened = image + alpha * mask

    # Clip the sharpened image to the valid range.
    sharpened.clamp_(0, 1)

    return sharpened


if __name__ == "__main__":
    image_path = r"C:\Users\GGPC\PycharmProjects\pythonProject\pyTorchLearning\Yolov8\bus.jpg"
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)

    # edge_extractor = EdgeExtractor()
    # image = edge_extractor(tensor_image)
    # image = enhance_edges(tensor_image)
    # final_image = image + tensor_image

    # blur, final_image = sharpen_image_with_unsharp_mask_torch(tensor_image)
    #
    # trans = transforms.ToPILImage()
    # blur = trans(blur.squeeze(0))
    # blur.show()

    image = sharpen_image_with_unsharp_mask_opti(tensor_image)
    trans = transforms.ToPILImage()
    image = trans(image.squeeze(0))
    image.show()
    # print(image)