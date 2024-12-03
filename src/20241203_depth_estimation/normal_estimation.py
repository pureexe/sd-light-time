import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
import torchvision.transforms.functional as F

# Initialize the depth estimation pipeline
depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

def compute_normal_map(depth_map):
    """
    Compute the normal map from a depth map.
    Depth map must be in OpenGL convention: Z-axis out, X to the right, Y up.
    """
    h, w = depth_map.shape
    normal_map = np.zeros((h, w, 3), dtype=np.float32)

    # Compute gradients
    dx = np.gradient(depth_map, axis=1)  # Gradient in x-direction
    dy = np.gradient(depth_map, axis=0)  # Gradient in y-direction

    # For each pixel, compute the normal vector
    for y in range(h):
        for x in range(w):
            dz_dx = dx[y, x]
            dz_dy = dy[y, x]

            # Normal vector in OpenGL convention
            #normal = np.array([-dz_dx, -dz_dy, 1.0])
            normal = np.array([-dz_dx, -dz_dy, 1.0])
            normal /= np.linalg.norm(normal)  # Normalize the vector

            normal_map[y, x] = normal

    return normal_map 

    # Normalize to [0, 1] range for visualization
    #return (normal_map + 1) / 2.0

def process_images(input_dir, output_dir):
    """
    Process all images in a directory, estimate depth maps,
    compute normal maps, and save them as .npy files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with tqdm(total=len(image_files), desc="Processing Images") as pbar:
        for file_name in image_files:
            input_path = os.path.join(input_dir, file_name)

            # Load and preprocess the image
            image = Image.open(input_path).convert("RGB")

            # Predict the depth map
            depth_result = depth_pipeline(image)
            depth_map = depth_result['predicted_depth'].float()
            depth_map = F.resize(depth_map, (512,512), antialias=True)
            depth_map = np.array(depth_map[0])
            
            # Compute the normal map
            normal_map = compute_normal_map(depth_map)

            # Save the normal map to a .npy file
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.npy")
            np.save(output_path, normal_map)

            pbar.update(1)

if __name__ == "__main__":
    input_directory = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/images"  # Replace with the directory of input images
    output_directory = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/normals"  # Replace with the directory to save output .npy files

    process_images(input_directory, output_directory)
