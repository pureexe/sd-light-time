import os
import json
import numpy as np
from tqdm.auto import tqdm

def compute_average_from_txt(folder_path):
    values = []
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                try:
                    values.extend([float(line.strip()) for line in file if line.strip()])
                except ValueError:
                    print(f"Skipping non-numeric content in {file_path}")
    return np.mean(values) if values else None

def main(root_dir, folders):
    scores = {}
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            avg_value = compute_average_from_txt(folder_path)
            if avg_value is not None:
                scores[folder] = avg_value
                print(f"Average for {folder}: {avg_value:.4f}")
            else:
                print(f"No valid numerical data found in {folder}")
        else:
            print(f"Folder not found: {folder_path}")
    
    # Save scores to a JSON file
    json_path = os.path.join(root_dir, "scores.json")
    with open(json_path, "w") as json_file:
        json.dump(scores, json_file, indent=4)
    print(f"Scores saved to {json_path}")

if __name__ == "__main__":
    root_dir = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_valid_face_same/default/1.0/newshading_newgt/1e-4/chk29/lightning_logs/version_102976"  # Change this to your root directory
    folders = ['psnr', 'mse', 'ddsim', 'ssim']
    main(root_dir, folders)