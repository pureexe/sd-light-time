import os
import ezexr
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiprocessing import Pool

SPLIT = "train"
PREDICT_DIR = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/{SPLIT}"
NUM_WORKERS = 20

def plot_and_save_histogram(data_list, bins=100, filename='histogram.png', name="Histogram"):
    """
    Plots a histogram from a list of arrays and saves it as a PNG file.
    """
    combined_data = np.array(data_list)
    plt.figure(figsize=(8, 6))
    plt.hist(combined_data, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

def process_exr(image_path):
    """ Reads an EXR file and computes relevant statistics. """
    pixel = ezexr.imread(image_path)
    return np.percentile(pixel, 99), np.percentile(pixel, 50), pixel.max()

def main():
    scenes = sorted(os.listdir(PREDICT_DIR))
    exr_files = []
    
    for scene in scenes:
        scene_root = os.path.join(PREDICT_DIR, scene, 'lightning_logs')
        versions = sorted(os.listdir(scene_root))
        exr_dir = os.path.join(scene_root, versions[-1], 'shading_exr')
        files = os.listdir(exr_dir)
        exr_files.extend(os.path.join(exr_dir, f) for f in files)
    
    percentile99 = []
    percentile50 = []
    max_values = []
    
    with Pool(NUM_WORKERS) as pool:
        for p99, p50, max_val in tqdm(pool.imap(process_exr, exr_files), total=len(exr_files)):
            percentile99.append(p99)
            percentile50.append(p50)
            max_values.append(max_val)
    print("MOST HIGHEST VALUE IS: ", np.array(max_values).max())
    plot_and_save_histogram(percentile99, filename="percentile99.png", name="percentile99")
    plot_and_save_histogram(percentile50, filename="percentile50.png", name="median")
    plot_and_save_histogram(max_values, filename="max_values.png", name="max values")

if __name__ == "__main__":
    main()
