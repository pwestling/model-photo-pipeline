import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import scipy.ndimage
import argparse

clicked_points = []
# Parse the command-line arguments
parser = argparse.ArgumentParser(
    description="Prepare an image of a model for display"
)
parser.add_argument(
    "input_path", help="Path to the original image"
)
parser.add_argument(
    "output_path", help="Location to write the final output image"
)
# Argument to show or not show intermediate debug image
parser.add_argument(
    "--debug", action="store_true", help="Show intermediate debug images"
)
parser.add_argument(
    "--no-cache", action="store_true", help="Don't cache the image embedding"
)
parser.add_argument(
    "--smooth-mask", action="store_true", help="Run dilation/erosion on the mask to fill holes"
)
parser.add_argument(
    "--device", default="cuda", help="Device to use for inference"
)
args = parser.parse_args()

# Get the filename
image_path = args.input_path
filename = os.path.basename(image_path)
image = Image.open(image_path)
width, height = image.size

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_file = f"mask-{filename}.npy"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Check if the checkpoint exists
if not os.path.exists(sam_checkpoint):
    print(
        f"Error: Checkpoint not found at {sam_checkpoint}. Download the checkpoint file using the link at https://github.com/facebookresearch/segment-anything#model-checkpoints ")
    exit()

device = args.device
print("Loading model...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

print("Creating predictor...")
predictor = SamPredictor(sam)

print("Setting image...")
predictor.set_image(image)
mask_image_handle = None

def show_mask(mask, ax, output_file=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if output_file is not None:
        mask_image_8bit = (mask_image * 255).astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image_8bit)
        mask_image_pil.save(output_file)
    global mask_image_handle
    mask_image_handle = ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Callback function for handling user clicks on the image
def on_click(event):
    if event.button == 1:  # Left mouse button
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        print(f"Clicked point: ({x}, {y})")
        print("Predicting...")
        input_point = np.array(clicked_points)
        input_label = np.array([1] * len(clicked_points))
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        if mask_image_handle is not None:
            mask_image_handle.remove()
        show_mask(masks[0], plt.gca(), random_color=False)
    elif event.button == 3:  # Right mouse button
        plt.close()

# Load the image
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

def compute_mask(image):
    # Display the image and collect clicked points
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    print("Left-click to select points. Right-click to finish.")
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    print("All clicked points:", clicked_points)

    sys.path.append("..")

    print("Predicting...")
    input_point = np.array(clicked_points)
    input_label = np.array([1] * len(clicked_points))
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    return masks[0]




if not args.no_cache and os.path.exists(mask_file):
    # Load the mask from file
    print("Loading mask from file...")
    best_mask = np.load(mask_file)
else:
    # Compute the mask
    print("Computing mask...")
    best_mask = compute_mask(image)
    # Save the mask to file
    np.save(mask_file, best_mask)

# Continue with the rest of the script

def adjust_saturation_contrast(image, mask, saturation_scale=1.1, contrast_scale=1.1):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase the saturation channel by the saturation_scale
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)

    # Convert the HSV image back to RGB
    adjusted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Increase the contrast of the adjusted image
    mean = np.mean(adjusted_image, axis=(0, 1), keepdims=True)
    adjusted_image = np.clip((adjusted_image - mean) * contrast_scale + mean, 0, 255).astype(np.uint8)

    # Blend the original image and the adjusted image using the mask
    result = (image * (1 - mask[..., None]) + adjusted_image * mask[..., None]).astype(np.uint8)

    return result

def fill_mask_holes(mask, kernel_size=5, iterations=2):

    mask = (mask * 255).astype(np.uint8) 
    # Create a structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation followed by erosion
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    filled_mask = cv2.erode(dilated_mask, kernel, iterations=iterations)
    filled_mask = filled_mask.astype(np.float32) / 255

    return filled_mask


def apply_gradient_background(image, mask, start_color=(0, 0, 0), end_color=(90, 90, 90), gradient_start_ratio=0.4):
    # Create a gradient background with the same size as the image
    background = np.zeros_like(image, dtype=np.float32)
    height, width, _ = image.shape

    # Find the highest 1 pixel in the mask
    highest_mask_pixel = np.where(mask == 1)[0].min()

    # Find the lowest 1 pixel in the mask
    lowest_mask_pixel = np.where(mask == 1)[0].max()

    # Calculate the gradient start position based on the ratio
    gradient_start = int(highest_mask_pixel + (height - highest_mask_pixel) * gradient_start_ratio)
    # End the gradient at the lowest mask pixel
    gradient_end = lowest_mask_pixel

    for i, color in enumerate(start_color):
        background[:gradient_start, :, i] = start_color[i]
        gradient = np.linspace(start_color[i], end_color[i], height - gradient_start, gradient_end).astype(np.float32)
        background[gradient_start:, :, i] = np.tile(gradient[:, np.newaxis], width)

    # Blend the image and the background using the mask
    blended_image = (image * mask[..., None] +
                     background * (1 - mask[..., None])).astype(np.uint8)

    return blended_image

def crop_image_with_mask(image, mask, buffer_ratio=0.25):
    # Calculate the center of mass of the mask
    print("Computing center of mass...")
    center_of_mass = scipy.ndimage.center_of_mass(mask)
    print("Cropping image...")

    # Find the bounding box coordinates for the mask
    rows, cols = np.where(mask)
    top, bottom = np.min(rows), np.max(rows)
    left, right = np.min(cols), np.max(cols)

    # Calculate the buffer size
    height_buffer = int((bottom - top) * buffer_ratio)
    width_buffer = int((right - left) * buffer_ratio)

    # Add half of the buffer size to the bounding box coordinates
    height = bottom - top
    width = right - left
    top = max(0, int(center_of_mass[0] - height // 2 - height_buffer // 2))
    bottom = min(image.shape[0], int(center_of_mass[0] + height // 2 + height_buffer // 2))
    left = max(0, int(center_of_mass[1] - width // 2 - width_buffer // 2))
    right = min(image.shape[1], int(center_of_mass[1] + width // 2 + width_buffer // 2))

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    return cropped_image, (top, bottom, left, right)

def save_and_display_image(image, name):
    if args.debug:
        image_pil = Image.fromarray(image)
        image_pil.save(name)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

if args.smooth_mask:
    best_mask = fill_mask_holes(best_mask)
    if args.debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(best_mask, plt.gca(), random_color=True)
        plt.title(f"Smoothed Mask", fontsize=18)
        plt.axis('off')
        plt.show()
saturated_image = adjust_saturation_contrast(image, best_mask)
save_and_display_image(saturated_image, 'saturated_image.png')
gradient_image = apply_gradient_background(saturated_image, best_mask)
save_and_display_image(gradient_image, 'gradient_image.png')
cropped_image, (top, bottom, left, right) = crop_image_with_mask(gradient_image, best_mask)
save_and_display_image(cropped_image, 'cropped_image.png')

# Save the cropped image to output file
cropped_image_pil = Image.fromarray(cropped_image)
cropped_image_pil.save(args.output_path)




