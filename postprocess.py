import os
import numpy as np

from PIL import Image

image_dir = "input_images"
result_dir = "output_images"
mask_dir = "mask_images"
final_dir = "final_images"


def detect_clothes(image_name):
    segmented_image_path = os.path.join(result_dir, image_name[:-3] + "png")
    original_image_path = os.path.join(image_dir, image_name[:-3] + "jpg")
    segmented_image = Image.open(segmented_image_path)
    original_image = Image.open(original_image_path)

    # Convert the image to a NumPy array
    segmented_array = np.array(segmented_image)
    original_array = np.array(original_image)

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
        masked_image.save(os.path.join(final_dir, f'{image_name[:-3]}_{i}.png'))


images_list = sorted(os.listdir(result_dir))
for image_name in images_list:
    # Load segmented image
    segmented_image_path = os.path.join(result_dir, image_name[:-3] + "png")
    original_image_path = os.path.join(image_dir, image_name[:-3] + "jpg")
    segmented_image = Image.open(segmented_image_path)
    original_image = Image.open(original_image_path)

    # Convert the image to a NumPy array
    segmented_array = np.array(segmented_image)
    original_array = np.array(original_image)

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
        mask_image.save(os.path.join(mask_dir, f'{image_name[:-3]}_mask_{i}.png'))

        # Save the mask applied image
        masked_image = Image.fromarray((255-mask_applied)%255, mode='RGB')
        masked_image.save(os.path.join(final_dir, f'{image_name[:-3]}_{i}.png'))
