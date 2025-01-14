import torch
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import os

def create_color_jitter_grid(image_path, grid_size=10, save_path=None):
    """
    Create a grid of images with color jitter applied.

    Parameters:
        image_path (str): Path to the input image.
        grid_size (int): Size of the grid (default is 10x10).
        save_path (str, optional): Path to save the grid image. If None, the image will only be displayed.

    Returns:
        None
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Define the ColorJitter transformation
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.05, hue=0.0)

    # Transform to tensor
    to_tensor = transforms.ToTensor()

    # List to store jittered images
    images = []

    for _ in range(grid_size * grid_size):
        jittered_image = color_jitter(image)
        images.append(to_tensor(jittered_image))

    # Create the grid
    grid = utils.make_grid(images, nrow=grid_size, padding=2)

    # Convert grid to numpy for displaying
    grid_np = grid.permute(1, 2, 0).numpy()

    # Plot the grid
    plt.figure(figsize=(grid_size*4, grid_size*4))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title("Color Jittered Image Grid")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

# Example usage
image_path = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10/dir_0_mip2.jpg"  # Replace with your image path
save_path = "color_jitter_grid.jpg"  # Optional: specify a save path

if os.path.exists(image_path):
    create_color_jitter_grid(image_path, grid_size=5, save_path=save_path)
else:
    print(f"Image file not found: {image_path}")
