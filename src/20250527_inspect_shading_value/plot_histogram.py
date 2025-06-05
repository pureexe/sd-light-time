import ezexr
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

def find_exr_files(root_dir):
    """Recursively find all .exr files under root_dir"""
    return list(Path(root_dir).rglob("*.exr"))

def compute_histogram_from_exr(file_path):
    """Read EXR and return flattened pixel values"""
    try:
        file_path = str(file_path)  # Ensure it's a string for ezexr
        img = ezexr.imread(file_path)  # shape: (H, W, C)
        img =  img.flatten()
        img = np.percentile(img, 99)  # Clip to 99.9% to avoid outliers
        img = np.array([img])
        return img
    except Exception as e:
        with open("error.log", "a") as log_file:
            log_file.write(f"{file_path} failed: {e}\n")
        return None

def main(root_dir, output_png="histogram.png", max_files=None):
    exr_files = find_exr_files(root_dir)
    if max_files:
        exr_files = exr_files[:max_files]

    print(f"Found {len(exr_files)} EXR files.")

    # Use multiprocessing Pool with tqdm
    with Pool(processes=24) as pool:
        results = list(tqdm(pool.imap(compute_histogram_from_exr, exr_files), total=len(exr_files), desc="Reading EXRs"))
    # for job in tqdm(exr_files, desc="Reading EXRs"):
    #     results = []
    #     try:
    #         result = compute_histogram_from_exr(job)
    #         results.append(result)
    #     except Exception as e:
    #         print(f"Error processing {job}: {e}")
    #         results.append(None)

    # Filter out failed reads (None)
    all_pixels = [r for r in results if r is not None]

    if not all_pixels:
        print("No valid EXR files were read.")
        return

    # Concatenate and compute histogram
    all_pixels = np.concatenate(all_pixels)
    print(f"Total pixels read: {len(all_pixels)}")
    print("max pixel value:", np.max(all_pixels))
    print("min pixel value:", np.min(all_pixels))
    print("mean pixel value:", np.mean(all_pixels))
    print("percentile 99.9% pixel value:", np.percentile(all_pixels, 99))
    plt.figure(figsize=(10, 6))
    plt.hist(all_pixels, bins=256, color='blue', alpha=0.7)
    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save to PNG
    plt.savefig(output_png)
    print(f"Histogram saved to {output_png}")

if __name__ == "__main__":
    main(root_dir="/pure/f1/datasets/multi_illumination/real_image_gt_shading/v0/train/albedos/",
         output_png="exr_gt_albedo.png")
