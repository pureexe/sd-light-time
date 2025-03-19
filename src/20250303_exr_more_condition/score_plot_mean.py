import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# def moving_average(data: list, weight: float) -> list:
#     """
#     Computes a moving average where the latest value has a given weight, 
#     and the rest share the remaining weight.
    
#     :param data: List of float values.
#     :param weight: Weight for the latest value (e.g., 0.9).
#     :return: List of moving averages.
#     """
#     if not data or not (0 < weight < 1):
#         return []

#     result = []
#     for i in range(1, len(data) + 1):
#         latest_weight = weight
#         rest_weight = (1 - weight) / (i - 1) if i > 1 else 0
#         avg = sum(data[j] * (latest_weight if j == i - 1 else rest_weight) for j in range(i))
#         result.append(avg)

#     return result

LEARNING_RATE='1e-4'
MOVING_SCALE = 0.5


def moving_average(data: list, alpha: float) -> list:
    data = np.asarray(data, dtype=np.float32)
    ema = np.empty_like(data)
    ema[0] = data[0]  # Initialize with the first data point

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema

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
        print("READING...: ", filepath)
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return np.mean(values) if values else None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_version_folder(path):
    """Get the latest version folder in a checkpoint directory."""
    try:
        all_dirs = sorted(os.listdir(path))
        return all_dirs[-1] if all_dirs else None
    except Exception as e:
        print(f"Error finding version folder in {path}: {e}")
        return None

def plot_mean_psnr(folders, output_file="mean_psnr_plot.png"):
    """Computes the mean PSNR across different folders and plots a single averaged line."""
    psnr_dict = {}  # Dictionary to store values by checkpoint number
    
    for folder_path in folders:
        split_path = folder_path.split("/chk")
        if len(split_path) < 2:
            print(f"Invalid path structure for {folder_path}")
            continue
        
        base_path = split_path[0]
        chk_nums = extract_checkpoints(base_path)
        
        for chk in chk_nums:
            chk_folder = os.path.join(base_path, f"chk{chk}", 'lightning_logs')
            version_dir = get_version_folder(chk_folder)
            if not version_dir:
                continue

            chk_folder = os.path.join(chk_folder, version_dir, 'psnr')
            psnr_files = glob.glob(os.path.join(chk_folder, "*.txt"))
            
            avg_values = [read_psnr_file(f) for f in psnr_files]
            avg_values = [v for v in avg_values if v is not None]
            
            if avg_values:
                if chk not in psnr_dict:
                    psnr_dict[chk] = []
                psnr_dict[chk].extend(avg_values)

    # Compute mean for each checkpoint
    sorted_checkpoints = sorted(psnr_dict.keys())
    mean_psnr_values = [np.mean(psnr_dict[chk]) for chk in sorted_checkpoints]
    
    # apply moving average
    moving_mean_psnr_values =  moving_average(mean_psnr_values, MOVING_SCALE)


    df = pd.DataFrame({'checkpoint': sorted_checkpoints, 'psnr': mean_psnr_values, f'moving_{MOVING_SCALE}': moving_mean_psnr_values})
    df.to_csv("mean_psnr_0_3_4_20.csv", index=False)

    # Plot the averaged PSNR curve
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_checkpoints[:-1], moving_mean_psnr_values[:-1], marker='o', linestyle='-', color='b', label="Mean PSNR")
    
    plt.xlabel("Checkpoint Number")
    plt.ylabel(f"Moving Average PSNR (Exp Moving AVG {MOVING_SCALE:.2f})")
    plt.title(f"LR: {LEARNING_RATE}, Mean PSNR (Exp Moving AVG {MOVING_SCALE:.2f}) vs Checkpoint")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

# Example usage
folders = [
    f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light0_exr_newgt/default/1.0/newshading_newgt/{LEARNING_RATE}/chk95/lightning_logs/version_101142/psnr",
    f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light3_exr_newgt/default/1.0/newshading_newgt/{LEARNING_RATE}/chk95/lightning_logs/version_101142/psnr",
    f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light4_exr_newgt/default/1.0/newshading_newgt/{LEARNING_RATE}/chk95/lightning_logs/version_101142/psnr",
    f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light20_exr_newgt/default/1.0/newshading_newgt/{LEARNING_RATE}/chk95/lightning_logs/version_101142/psnr",
]

plot_mean_psnr(folders, output_file=f"lr{LEARNING_RATE}_mean{MOVING_SCALE}_psnr_0_3_4_20.png")