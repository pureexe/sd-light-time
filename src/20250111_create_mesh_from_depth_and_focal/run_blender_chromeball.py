import os 
import json 
from tqdm.auto import tqdm 
import numpy as np
import math
import argparse

BLENDER_PATH = "/home/pakkapon/mnt_tl_vision23/home/vll/software/blender-3.6.5-linux-x64/blender"
ENVMAP_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/unused/exr_envmap_train_mip2_exr_v3"
OUTPUT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/viz_chromeball_blender_mesh_perspective_v2"
OBJ_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/mesh/"
JSON_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/focal_json/"
IMAGE_WIDTH = 512

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, default=0)
parser.add_argument('-t', '--total', type=int, default=1)
args = parser.parse_args()



def print_error():
    ascii_art = """
  _____  ____  ____  _____  ____  
 | ____||  _ \|  _ \| ____||  _ \ 
 |  _|  | | | | | | |  _|  | | | |
 | |___ | |_| | |_| | |___ | |_| |
 |_____||____/|____/|_____||____/ 
    """
    print(ascii_art)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenes = os.listdir(JSON_PATH)
    scenes = [f.replace(".json","") for f in sorted(scenes)]
    queues = []
    for scene in scenes[:1]:
        for light_id in range(25):
            queues.append([scene, light_id])
    queues = queues[args.index::args.total]

    for info in tqdm(queues):
        scene, light_id = info
        out_path = f"{OUTPUT_DIR}/{scene}/dir_{light_id}_mip2.png"
        if os.path.exists(out_path):
            continue
        # read json
        json_path = f"{JSON_PATH}/{scene}.json"
        try:
            with open(json_path,'r') as f: 
                data = json.load(f)
                focal = data['focal']
                z_offset = data['z_offset']
        except:
            print("---------------------------------------------------------------------")
            print_error()
            print("---------------------------------------------------------------------")
            print(f"Error: {scene}/{light_id} - {json_path}")
            print("---------------------------------------------------------------------")
            continue

        fov_rad = 2 * np.arctan2(IMAGE_WIDTH, 2*focal)
        #fov_deg = fov_rad / np.pi * 180
        os.makedirs(f"{OUTPUT_DIR}/{scene}",exist_ok=True)
        cmd = f"{BLENDER_PATH} -b -P blender_render_chromeball.py -- {OBJ_PATH}/{scene}.obj {ENVMAP_PATH}/{scene}/probes/dir_{light_id}_chrome256.exr {fov_rad:.8f} {z_offset:.8f} {out_path}"
        os.system(cmd)
        
if __name__ == "__main__":
    main()