import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import gradio as gr

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

device = "cpu"

image_dir = "input_images"
result_dir = "output_images"
mask_dir = "mask_images"
final_dir = "final_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm.pth")
do_palette = True


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
    #original_image_path = os.path.join(image_dir, image_name[:-3] + "jpg")
    segmented_image = Image.open(segmented_image_path)
    #original_image = Image.open(original_image_path)
    original_image = image_name

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
    
    #img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    img = image_name
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
    output_img.save(os.path.join(result_dir, "segmented." + "png"))

    detect_clothes(image_name)

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
    
    ret_list = [f'{final_dir}/{i}' for i in os.listdir(final_dir)]
    
    return ret_list


def clear_folders():
    def remove_files_in_folder(folder_path):
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Error while removing files in {folder_path}: {e}")

    # Folder paths
    folders = ["mask_images", "output_images", "final_images"]

    # Remove files in each folder
    for folder in folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Removing files in folder: {folder}")
            remove_files_in_folder(folder_path)
        else:
            print(f"Folder does not exist: {folder}")

    print("All files removed successfully.")

def demo_interface():
    demo = gr.Blocks()

    with demo:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type='numpy', label="Original Image", interactive=True)
                with gr.Row():
                    find_button = gr.Button('Find')

            with gr.Column():
                clothes = gr.Gallery(label='Clothes', height='auto')
            
            with gr.Column():
                with gr.Row():
                    remove_files_button = gr.Button('Clear Files')
                clear_info = gr.Textbox(label="Clear Files Info")
    
            """    with gr.Row():
                    drop_collection_button = gr.Button('Drop Collection')
                drop_collection_info = gr.Textbox(label='Drop Collection Info')

                with gr.Row():
                    # create a checkbox to choose which dataset to search
                    dataset_checkbox = gr.CheckboxGroup(
                        label="Dataset",
                        choices=[ds_name['name'] for ds_name in dataset_names],
                    )
                
                with gr.Row():
                    read_embeddings_button = gr.Button('Read Embeddings')
                with gr.Row():
                    read_embeddings_info = gr.Textbox(label="Embedding read time")
                    len_embeddings_info = gr.Textbox(label="Embedding read time")

                with gr.Row():
                    load_model_button = gr.Button('Load Model')
                with gr.Row():
                    # load_model_label = gr.Label(label="Load model")
                    load_model_info = gr.Textbox(label="Load model info")

                with gr.Row():
                    create_db_button = gr.Button('Create DB')
                with gr.Row():
                    # create_db_label = gr.Label(label="Create DB")
                    create_db_info = gr.Textbox(label="Create DB info")

                with gr.Row():
                    upload_data_button = gr.Button('Upload Data')
                upload_data_info = gr.Textbox(label="Upload data info")

                with gr.Row():
                    free_memory_button = gr.Button('Free embeddings memory')
                free_memory_info = gr.Textbox(label="Upload data info")"""
        
        find_button.click(
            segmentate_img, # done
            inputs=[input_image],
            outputs=[clothes]
        )
        remove_files_button.click(
            clear_folders,
            outputs=[clear_info]
        )

        """get_info_button.click(
            get_collection_info, # done
            outputs=[collection_info]
        )
        drop_collection_button.click(
            drop_collection_and_db, # done
            outputs=[drop_collection_info]
        )
        read_embeddings_button.click(
            get_embeddings, # done
            inputs=[dataset_checkbox],
            outputs=[read_embeddings_info, len_embeddings_info]
        )
        load_model_button.click(
            get_model_and_transform, # done
            outputs=[load_model_info]
        )
        create_db_button.click(
            get_db, # done
            outputs=[create_db_info]
        )
        upload_data_button.click(
            upload_data, # done
            outputs=[upload_data_info]
        )
        free_memory_button.click(
            free_memory, # done
            outputs=[free_memory_info]
        )"""

        demo.launch(share=True)

if __name__ == '__main__':
    demo_interface()