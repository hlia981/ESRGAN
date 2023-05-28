import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from datasets import denormalize
from models import GeneratorRRDB
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", type=str, required=True, help="Path to image")
# parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
# parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
# parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
# opt = parser.parse_args()
# print(opt)
image_path = r"C:\Users\hlia981\mmdetection\ESRGAN\images\training\63700.png"
checkpoint_model = r"C:\Users\hlia981\mmdetection\ESRGAN\saved_models\generator_53.pth"
channels = 3
residual_blocks = 23

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(channels, filters=64, num_res_blocks=residual_blocks).to(device)
generator.load_state_dict(torch.load(checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)

# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor)).cpu()

# Save image
fn = image_path.split("/")[-1]
save_image(sr_image, f"images/outputs/sr-{fn}")