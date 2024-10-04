# This is the altered version of the infer code, written by Esad

import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

import cv2

device = "cuda"

image_dir = "input_images"
result_dir = "output_images"
mask_dir = "mask_images"
final_dir = "final_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm.pth")
do_palette = True

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

create_directory_if_not_exists(image_dir)
create_directory_if_not_exists(result_dir)
create_directory_if_not_exists(mask_dir)
create_directory_if_not_exists(final_dir)


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def detect_clothes(image_name):
    segmented_image_path = os.path.join(result_dir, "segmented." + "png")
    original_image_path = os.path.join(image_dir, image_name[:-3] + "jpg")
    segmented_image = Image.open(segmented_image_path)
    original_image = Image.open(original_image_path)

    # Convert the image to a NumPy array
    segmented_array = np.array(segmented_image)
    #original_array = np.array(original_image)
    original_array = original_image

    # Find unique colors
    unique_colors = np.unique(segmented_array)

    for i, color in enumerate(unique_colors):
        if color == 0: # Skip the black color
            continue
        # Create a binary mask for the current color
        binary_mask = segmented_array == color

        # Convert binary mask to uint8 for saving as an image
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

        # Apply mask to the original image
        mask_applied = np.zeros_like(original_array).astype(np.uint8)
        for j in range(original_array.shape[2]):
            mask_applied[:, :, j] = np.multiply(original_array.astype(np.uint8)[:, :, j], binary_mask_uint8)

        # Save the mask applied image
        masked_image = Image.fromarray((255-mask_applied)%255, mode='RGB')
        masked_image.save(os.path.join(final_dir, f'final_{i}.png'))


def segmentate_img(image_name):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    palette = get_palette(4)
    
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
        output_img.putpalette(palette)
    output_img.save(os.path.join(result_dir, f"{image_name[:-3]}segmented.png"))

    # Segmentation done here and image saved to result_dir

    detect_clothes(image_name)

    """
    # Convert the image to a NumPy array
    segmented_array = output_arr
    original_array = image_name

    # Find unique colors
    unique_colors = np.unique(segmented_array)

    for i, color in enumerate(unique_colors):
        if color == 0: # Skip the black color
            continue
        # Create a binary mask for the current color
        binary_mask = segmented_array == color

        # Convert binary mask to uint8 for saving as an image
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

        # Apply mask to the original image
        mask_applied = np.zeros_like(original_array).astype(np.uint8)
        for j in range(original_array.shape[2]):
            mask_applied[:, :, j] = np.multiply(original_array.astype(np.uint8)[:, :, j], binary_mask_uint8)

        # Save the binary mask as an image
        mask_image = Image.fromarray(binary_mask_uint8)
        mask_image.save(os.path.join(mask_dir, f'mask_{i}.png'))

        # Save the mask applied image
        inverted_image = (255 - (mask_applied))%255
        
        true_black_background = np.all(mask_applied == [0, 0, 0], axis=-1)

        # Convert these true black background pixels to white in the inverted image
        inverted_image[true_black_background] = [255, 255, 255]

        masked_image = Image.fromarray(inverted_image, mode='RGB')
        masked_image.save(os.path.join(final_dir, f'final_{i}.png'))
    """

if __name__ == '__main__':
    image_name = "deneme.png"
    segmentate_img(image_name)