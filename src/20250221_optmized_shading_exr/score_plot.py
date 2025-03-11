import os
import re
print("Importing numpy...")
import numpy as np
print("importing matplotlib...")
import matplotlib.pyplot as plt
import glob
import pandas as pd
print("importing done")

def extract_checkpoints(base_path):
    """Extracts all available checkpoints in the given base path."""
    try:
        chk_dirs = [d for d in os.listdir(base_path) if re.match(r'chk\d+', d)]
        chk_nums = sorted([int(d[3:]) for d in chk_dirs])
        return chk_nums
    except Exception as e:
        print(f"Error listing checkpoints in {base_path}: {e}")
        return []

def read_psnr_file(filepath):
    """Reads a PSNR file and computes the average value."""
    try:
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return np.mean(values) if values else None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_version_folder(path):
    all_dirs = sorted(os.listdir(path))
    return all_dirs[-1]

def plot_psnr(folders, labels, output_file="psnr_plot.png"):
    """Reads PSNR data from multiple folders, extracts checkpoints, averages values, and plots them."""
    plt.figure(figsize=(8, 5))
    
    pd_dict = {}
    for folder_path, label in zip(folders, labels):
        split_path = folder_path.split("/chk")
        if len(split_path) < 2:
            print(f"Invalid path structure for {folder_path}")
            continue
        
        base_path = split_path[0]
        
        chk_nums = extract_checkpoints(base_path)
        
        data = []
        for chk in chk_nums:
            chk_folder = os.path.join(base_path, f"chk{chk}", 'lightning_logs')
            version_dir = get_version_folder(chk_folder)
            chk_folder = os.path.join(chk_folder,version_dir, 'psnr')
            print(chk_folder)
            psnr_files = glob.glob(os.path.join(chk_folder, "*.txt"))
            
            avg_values = [read_psnr_file(f) for f in psnr_files]
            avg_values = [v for v in avg_values if v is not None]
            
            if avg_values:
                data.append((chk, np.mean(avg_values)))
        
        if not data:
            print(f"No valid data found for {label}.")
            continue
        
        data.sort()  # Sort by checkpoint number
        x_values, y_values = zip(*data)
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=label)
        pd_dict[label+'_chk'] = x_values
        pd_dict[label+'_psnr'] = y_values
    
    df = pd.DataFrame(pd_dict)
    df.to_csv("psnr_0_3_4_20.csv", index=False)

    plt.xlabel("Checkpoint Number")
    plt.ylabel("Average PSNR")
    plt.title("PSNR vs Checkpoint")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")# Example usage

folders = [
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100430/epoch_0051/psnr",
]
labels = [
    "1e-4",
    "5e-5",
    "1e-5",
    "5e-6",
    "1e-6"
]
plot_psnr(folders, labels, output_file="learning_rate.png")
