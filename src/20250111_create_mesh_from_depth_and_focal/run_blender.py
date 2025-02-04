import os 
import json 
from tqdm.auto import tqdm 
import numpy as np
import math

BLENDER_PATH = "/home/pakkapon/mnt_tl_vision17/home/vll/software/blender-3.6.5-linux-x64/blender"
ENVMAP_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/unused/exr_envmap_train_mip2_exr_v2"
OUTPUT_DIR = "/home/pakkapon/mnt_tl_vision17/data/pakkapon/datasets/multi_illumination/spherical/train/control_shading_blender_mesh"
OBJ_PATH = "/home/pakkapon/mnt_tl_vision17/data/pakkapon/datasets/multi_illumination/spherical/train/mesh/"
JSON_PATH = "/home/pakkapon/mnt_tl_vision17/data/pakkapon/datasets/multi_illumination/spherical/train/focal_json/"
IMAGE_WIDTH = 512

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenes = os.listdir(JSON_PATH)
    scenes = [f.replace(".json","") for f in sorted(scenes)]
    queues = []
    for scene in scenes[14::15]:
        for light_id in range(25):
            queues.append([scene, light_id])

    for info in tqdm(queues):
        scene, light_id = info
        out_path = f"{OUTPUT_DIR}/{scene}/dir_{light_id}_mip2.png"
        if os.path.exists(out_path):
            continue
        # read json
        with open(f"{JSON_PATH}/{scene}.json",'r') as f: 
            data = json.load(f)
            focal = data['focal']
            z_offset = data['z_offset']

        fov_rad = 2 * np.arctan2(IMAGE_WIDTH, 2*focal)
        #fov_deg = fov_rad / np.pi * 180
        os.makedirs(f"{OUTPUT_DIR}/{scene}",exist_ok=True)
        cmd = f"{BLENDER_PATH} -b -P blender_render.py -- {OBJ_PATH}/{scene}.obj {ENVMAP_PATH}/{scene}/probes/dir_{light_id}_chrome256.exr {fov_rad:.8f} {z_offset:.8f} {out_path}"
        os.system(cmd)
        
if __name__ == "__main__":
    main()