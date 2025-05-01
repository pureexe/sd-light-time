import os
import json
import multiprocessing as mp
from tqdm import tqdm
import imageio.v3 as iio
import numpy as np
import argparse


def find_max_in_exr(file_path):
    try:
        img = iio.imread(file_path, extension=".exr")
        return float(np.max(img))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return float('-inf')


def get_all_exr_files_recursive(directory):
    exr_files = []
    for root, _, files in os.walk(directory, followlinks=True):
        for file in files:
            if file.lower().endswith(".exr"):
                exr_files.append(os.path.join(root, file))
    return exr_files


def main(input_dir, output_json):
    exr_files = get_all_exr_files_recursive(input_dir)

    if not exr_files:
        print("No EXR files found in the input directory.")
        return

    current_max = mp.Value('d', float('-inf'))
    lock = mp.Lock()

    with tqdm(total=len(exr_files), desc="Processing EXRs") as pbar:
        def update(result):
            with lock:
                if result > current_max.value:
                    current_max.value = result
            pbar.set_postfix_str(f"max: {current_max.value:.5f}")
            pbar.update()

        with mp.Pool(processes=8) as pool:
            for file_path in exr_files:
                pool.apply_async(find_max_in_exr, args=(file_path,), callback=update)
            pool.close()
            pool.join()

    output_data = {"shadings": {"max_value": current_max.value}}
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Max value saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", help="Directory containing EXR images")
    parser.add_argument("output_json_name", help="Path to output JSON file")
    args = parser.parse_args()

    main(args.input_directory, args.output_json_name)