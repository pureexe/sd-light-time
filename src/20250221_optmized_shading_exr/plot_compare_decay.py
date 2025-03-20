import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

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

def plot_chart(folders, names, folders2, names2, checkpoints, checkpoints2, checkpoint_start=160, output_file="plot_decay.png"):
    assert len(folders) == len(names)
    
    plt.figure(figsize=(8, 5))
    
    
    for idx in range(len(folders)):
        folder = folders[idx]
        name = names[idx]
        psnr_dict = {}
        for folder_path in folder:
            for chk_id in checkpoints:
                chk_folder = os.path.join(folder_path, f'chk{chk_id}', 'lightning_logs')
                if not os.path.exists(chk_folder):
                    continue 
                version_dir = get_version_folder(chk_folder)
                chk_folder = os.path.join(chk_folder, version_dir, 'psnr')
                psnr_files = glob.glob(os.path.join(chk_folder, "*.txt"))
                avg_values = [read_psnr_file(f) for f in psnr_files]
                avg_values = [v for v in avg_values if v is not None]
                if avg_values:
                    if chk_id not in psnr_dict:
                        psnr_dict[chk_id] = []
                    psnr_dict[chk_id].extend(avg_values)
        # Compute mean for each checkpoint
        sorted_checkpoints = sorted(psnr_dict.keys())
        mean_psnr_values = [np.mean(psnr_dict[chk]) for chk in sorted_checkpoints]
        #mean_psnr_values =  moving_average(mean_psnr_values, MOVING_SCALE)
        plt.plot(sorted_checkpoints[:-1], mean_psnr_values[:-1],  label=name, marker='o')
        
    ###### 
    print("==================================")
    print("==================================")
    print("==================================")
    for idx in range(len(folders2)):
        name = names2[idx]
        psnr_dict = {}
        for idy in range(len(folders2[idx])):
            folder_path = folders2[idx][idy] 
            for chk_id in checkpoints2:
                chk_folder = os.path.join(folder_path, f'epoch_{chk_id:04d}')
                if not os.path.exists(chk_folder):
                    continue
                chk_folder = os.path.join(chk_folder, 'psnr')
                psnr_files = glob.glob(os.path.join(chk_folder, "*.txt"))
                avg_values = [read_psnr_file(f) for f in psnr_files]
                avg_values = [v for v in avg_values if v is not None]
                if avg_values:
                    if chk_id not in psnr_dict:
                        psnr_dict[chk_id + checkpoint_start] = []
                    psnr_dict[chk_id + checkpoint_start].extend(avg_values)
        #Compute mean for each checkpoint
        sorted_checkpoints = sorted(psnr_dict.keys())
        mean_psnr_values = [np.mean(psnr_dict[chk]) for chk in sorted_checkpoints]
        #mean_psnr_values =  moving_average(mean_psnr_values, MOVING_SCALE)
        plt.plot(sorted_checkpoints[:-1], mean_psnr_values[:-1],  label=name, marker='o')
        
           
        
    plt.xlabel("Checkpoint ")
    plt.ylabel(f"PSNR ")
    plt.title(f"PSNR on different learning rate")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")



             

    
def main():
    folders = [
        [
            f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light0_exr_newgt/default/1.0/newshading_newgt/1e-4",
            f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light3_exr_newgt/default/1.0/newshading_newgt/1e-4",
            f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light4_exr_newgt/default/1.0/newshading_newgt/1e-4",
            f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/val_v3_all_14n_copyroom10_light20_exr_newgt/default/1.0/newshading_newgt/1e-4",            
        ]
    ]
    names = ['1e-4']
    folders2 = [
        [
            '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102596',
            '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_103073',            
        ],
        [
            '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102595',
            '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_103286',
        ]
    ]
    names2 = [
        'decay_5e-5',
        'decay_1e-5',
    ]
    checkpoints = range(150, 225)
    checkpoints2 = range(1, 60)
    checkpoint_start = 160
    plot_chart(folders, names, folders2, names2, checkpoints, checkpoints2, checkpoint_start)

if __name__ == "__main__":
    main()
    
