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
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".exr"):
                exr_files.append(os.path.join(root, file))
    return exr_files


def main(input_dir, output_json):
    exr_files = get_all_exr_files_recursive(input_dir)

    if not exr_files:
        print("No EXR files found in the input directory.")
        return

    with mp.Pool(processes=8) as pool:
        max_values = list(tqdm(pool.imap(find_max_in_exr, exr_files), total=len(exr_files)))

    global_max = max(max_values)
    output_data = {"shadings": {"max_value": global_max}}

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Max value saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", help="Directory containing EXR images")
    parser.add_argument("output_json_name", help="Path to output JSON file")
    args = parser.parse_args()

    main(args.input_directory, args.output_json_name)

    