# counter file 
import os 
from tqdm.auto import tqdm
import skimage
from multiprocessing import Pool
import numpy as np
import json 
from LineNotify import LineNotify
#CHK_PTS = [19, 39, 59, 79, 99, 119]
CHK_PTS = [34]
LRS = ['1e-4', '5e-4', '1e-4']
NAMES = ['vae_r2_g0', 'vae_r2_g0', 'vae_r2']
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']


def get_axis(val_dir):
    if "_x_" in val_dir:
        return 1
    elif "_y_" in val_dir:
        return 2
    elif "_z_" in val_dir:
        return 3
    else:
        raise ValueError("Invalid val_dir")

def convert_to_grayscale(v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        assert v.shape[0] == 3
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]


def process_job(meta):
    input_dir = meta["in_dir"]
    output_dir = meta["out_dir"]
    expected_axis = meta["expected_axis"]
    is_less = meta["is_less"]
    try:
        files = sorted(os.listdir(input_dir))
    except FileNotFoundError:
        return
    EXT = "_light.npy"
    files = [f for f in files if f.endswith(EXT)]
    outputs = {
        'corrected': [],
        'failed': []
    }
    for filename in files:
        light = np.load(os.path.join(input_dir, filename)) #shape (9,3)
        # permute numpy array to (3,9) instead
        light = light.T
        # convert to grayscale
        light = convert_to_grayscale(light)
        fname = filename.replace(EXT, "")
        if is_less and light[expected_axis] < 0:
            outputs['corrected'].append(fname)
        elif (not is_less) and light[expected_axis] > 0:
            outputs['corrected'].append(fname)
        else:
            outputs['failed'].append(fname)
    # save json file 
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(outputs, f, indent=4)

def main():
    process_files = []
    for version in range(2,3):
        lr = LRS[version]
        name = NAMES[version]
        for chk_pt in CHK_PTS:
            #print(f"Processing {name} {lr} {chk_pt}")
            for val_dir in VAL_FILES:
                dir_path = f"output/20240726/val_axis/{name}/{lr}/chk{chk_pt}/{val_dir}/lightning_logs/version_0/"
                in_dir = os.path.join(dir_path, "face_light")
                out_dir = dir_path
                try:
                    files = sorted(os.listdir(in_dir))
                except FileNotFoundError:
                    continue
                process_files.append({
                    "in_dir": in_dir,
                    "out_dir": out_dir,
                    "expected_axis": get_axis(val_dir),
                    "is_less": "minus" in val_dir,
                })
    
    
    with Pool(32) as p:
      r = list(tqdm(p.imap(process_job, process_files), total=len(process_files)))
    LineNotify().send("counter_axis2.py finished successfully", with_hostname=True)



if __name__ == "__main__":
    main()