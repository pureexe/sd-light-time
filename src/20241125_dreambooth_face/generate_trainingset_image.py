print("LOADING LIBRARY...")
import os
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import json
print("FINISHED LOADED LIBRARY")

# Define the image processing function
def process_images(image_paths, output_folder, grid_size=10, image_size=(256, 256)):
    """
    Create grids of images from the provided image paths.

    Args:
        image_paths (list): List of image paths to process.
        output_folder (str): Folder to save the output grids.
        grid_size (int): Number of rows and columns in each grid.
        image_size (tuple): Target size for resizing images.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define transformations for resizing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Number of images per grid
    images_per_grid = grid_size * grid_size

    # Process images in batches
    for grid_index in range(0, len(image_paths), images_per_grid):
        # Create a list to store the tensors
        image_tensors = []

        # Process each image in the current batch
        for image_path in tqdm(image_paths[grid_index:grid_index + images_per_grid], desc=f"Processing grid {grid_index // images_per_grid}"):
            try:
                # Open and transform the image
                img = Image.open(image_path).convert("RGB")
                img_tensor = transform(img)
                image_tensors.append(img_tensor)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Create a grid if there are enough images
        if image_tensors:
            grid = make_grid(torch.stack(image_tensors), nrow=grid_size, padding=2)
            grid_image = transforms.ToPILImage()(grid)

            # Save the grid image
            output_path = os.path.join(output_folder, f"{grid_index // images_per_grid:02}.jpg")
            grid_image.save(output_path)
            print(f"Saved grid to {output_path}")

def main():
    DIR = "/ist/ist-share/vision/relight/datasets/face/simple_face_lora/face_right_321"
    
    with open(os.path.join(DIR, 'face_right_321.json')) as f:
        data = json.load(f)

    image_paths = [os.path.join(DIR, "images", img +".jpg") for img in data['image_index']][:300]

    # Output folder for the grids
    output_folder = "training_set_321"
    
    # Run the process
    process_images(image_paths, output_folder)




if __name__ == "__main__":
    main()