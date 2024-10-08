import os
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def process_image(args):
    source_path, target_path, output_path, threshold = args

    # Read the source and target images
    source_image = imread(source_path)
    target_image = imread(target_path)

    # Ensure the images have the same shape
    if source_image.shape == target_image.shape:
        # Calculate absolute difference between images
        diff = np.abs(source_image.astype(np.float32) - target_image.astype(np.float32))

        # If image has multiple channels (e.g., RGB), reduce to 1 channel by checking any channel
        if len(diff.shape) == 3:  # If the image has 3 channels (e.g., RGB)
            # Create a mask where any of the channels exceed the threshold
            mask = np.any(diff > threshold, axis=-1)
        else:
            # For single-channel (grayscale) images, apply the threshold directly
            mask = diff > threshold

        # Convert mask to uint8 format (0 and 255)
        binary_mask = img_as_ubyte(mask)

        # Save the binary mask to the output directory as a single-channel image
        imsave(output_path, binary_mask)
    else:
        print(f"Image shape mismatch for: {os.path.basename(source_path)}")

def generate_difference_mask(source_dir, target_dir, output_dir, threshold):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of all files in the source directory
    source_files = os.listdir(source_dir)
    
    # Prepare a list of arguments to process images in parallel
    tasks = []
    for file_name in source_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Check if the corresponding file exists in the target directory
        if os.path.exists(target_path):
            tasks.append((source_path, target_path, output_path, threshold))
        else:
            print(f"Target image not found for: {file_name}")
    
    # Use multiprocessing with a pool of 8 threads
    with Pool(processes=8) as pool:
        # Use tqdm to show progress bar
        for _ in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks)):
            pass

# Example usage:
# generate_difference_mask('source_folder', 'target_folder', 'output_folder', 10)

def bothway():
    THRESHOLD = 10
    scene_dirs = []
    checkpoints = {
        'no_control': '254',
        'depth': '299',
        'bae': '304',
        'both_bae': '349'
    }
    for guidance_scale in [1.0, 3.0, 5.0, 7.0]:
        for method in ['bae', 'both_bae', 'depth', 'no_control']:
            checkpoint = checkpoints[method]
            for inversion_step in [5,10,25,50,100,200,250,500,999]:
                scene = f"output/20240929/val_multillum_ddim_bothway_guidance_val_array_v2/vae/{guidance_scale}/{method}/1e-4/chk{checkpoint}/inversion{inversion_step}/lightning_logs/version_0"
                if os.path.exists(scene):
                    scene_dirs.append(scene)
    for scene_dir in scene_dirs:
        source_dir = os.path.join(scene_dir, 'crop_image')
        target_dir = os.path.join(scene_dir, 'source_image')
        output_dir = os.path.join(scene_dir, 'diff_mask')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating difference mask for: {scene_dir}")
        generate_difference_mask(source_dir, target_dir, output_dir, THRESHOLD)

def train2():
    THRESHOLD = 10
    scene_dirs = []
    checkpoints = {
        'no_control': '254',
        'depth': '299',
        'bae': '304',
        'both_bae': '349'
    }
    for guidance_scale in [1.0, 3.0, 5.0, 7.0]:
        for method in ['bae', 'both_bae', 'depth', 'no_control']:
            checkpoint = checkpoints[method]
            for inversion_step in [5,10,25,50,100,200,250,500,999]:
                scene = f"output/20240929/val_multillum_train2_relight/vae/{guidance_scale}/{method}/1e-4/chk{checkpoint}/inversion{inversion_step}/lightning_logs/version_0"
                if os.path.exists(scene):
                    scene_dirs.append(scene)
    for scene_dir in scene_dirs:
        source_dir = os.path.join(scene_dir, 'crop_image')
        target_dir = os.path.join(scene_dir, 'source_image')
        output_dir = os.path.join(scene_dir, 'diff_mask')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating difference mask for: {scene_dir}")
        generate_difference_mask(source_dir, target_dir, output_dir, THRESHOLD)



def train2():
    THRESHOLD = 10
    scene_dirs = []
    checkpoints = {
        'no_control': '254',
        'depth': '299',
        'bae': '304',
        'both_bae': '349'
    }
    for guidance_scale in [1.0, 3.0, 5.0, 7.0]:
        for method in ['bae', 'both_bae', 'depth', 'no_control']:
            checkpoint = checkpoints[method]
            for inversion_step in [5,10,25,50,100,200,250,500,999]:
                scene = f"output/20240929/val_multillum_train2_relight/vae/{guidance_scale}/{method}/1e-4/chk{checkpoint}/inversion{inversion_step}/lightning_logs/version_0"
                if os.path.exists(scene):
                    scene_dirs.append(scene)
    for scene_dir in scene_dirs:
        source_dir = os.path.join(scene_dir, 'crop_image')
        target_dir = os.path.join(scene_dir, 'source_image')
        output_dir = os.path.join(scene_dir, 'diff_mask')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating difference mask for: {scene_dir}")
        generate_difference_mask(source_dir, target_dir, output_dir, THRESHOLD)

def train2_nulltext():
    THRESHOLD = 10
    scene_dirs = []
    checkpoints = {
        'no_control': '254',
        'depth': '299',
        'bae': '304',
        'both_bae': '349'
    }
    #output/20240929/val_multillum_train2_nulltext
    for guidance_scale in [2.0, 2.5, 3.0, 7.0]:
        for method in ['bae', 'both_bae', 'depth', 'no_control']:
            checkpoint = checkpoints[method]
            for inversion_step in [5,10,25,50,100,200,250,500,999]:
                scene = f"output/20240929/val_multillum_train2_nulltext/vae/{guidance_scale}/{method}/1e-4/chk{checkpoint}/lightning_logs/version_0"
                if os.path.exists(scene):
                    scene_dirs.append(scene)
    for scene_dir in scene_dirs:
        source_dir = os.path.join(scene_dir, 'crop_image')
        target_dir = os.path.join(scene_dir, 'source_image')
        output_dir = os.path.join(scene_dir, 'diff_mask')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating difference mask for: {scene_dir}")
        generate_difference_mask(source_dir, target_dir, output_dir, THRESHOLD)



def val_multilumn():
    THRESHOLD = 10
    scene_dirs = []
    checkpoints = {
        'no_control': '254',
        'depth': '299',
        'bae': '304',
        'both_bae': '349'
    }
    for guidance_scale in [2.0, 2.5, 3.0, 7.0]:
        for method in ['bae', 'both_bae', 'depth', 'no_control']:
            checkpoint = checkpoints[method]
            scene = f"output/20240929/val_multillum_test_30_array_v2/vae/{guidance_scale}/{method}/1e-4/chk{checkpoint}/lightning_logs/version_0"
            if os.path.exists(scene):
                scene_dirs.append(scene)
    for scene_dir in scene_dirs:
        source_dir = os.path.join(scene_dir, 'crop_image')
        target_dir = os.path.join(scene_dir, 'source_image')
        output_dir = os.path.join(scene_dir, 'diff_mask')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating difference mask for: {scene_dir}")
        generate_difference_mask(source_dir, target_dir, output_dir, THRESHOLD)


def main():
    #bothway()
    #train2()
    #val_multilumn()
    train2_nulltext()

    

if __name__ == "__main__":
    main()  