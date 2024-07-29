import os

import skimage.io 
from sphere_helper import drawSphere, drawMap
from LineNotify import notify
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm
import skimage
import torch

ROOT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/20240703/val_rotate_envmap/{}/chk{}/{}/lightning_logs/version_0/"
BALL_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/20240703/val_rotate_envmap/{}/chk{}/{}/lightning_logs/version_0/{}/{}_ball"
LIGHT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/20240703/val_rotate_envmap/{}/chk{}/{}/lightning_logs/version_0/{}/{}_light"


CHECKPOINTS = [ 179]
IMAGE_TYPE = ['light_x_minus','light_x_plus','light_y_minus', 'light_y_plus','light_z_minus', 'light_z_plus']
VERSIONS = ['vae5000/3e-4']

def process_file(data):
    #version, checkpoint, light_direction, filename = data
    version, checkpoint, light_direction, scene, axis, filename = data
    ball_dir = BALL_DIR.format(version, checkpoint, light_direction, scene, axis)
    # make directory if not exist
    os.makedirs(ball_dir, exist_ok=True)
    
    #envmap_dir = ENVMAP_DIR.format(version, checkpoint, light_direction, scene, axis)
    #os.makedirs(envmap_dir, exist_ok=True)

    light_dir = LIGHT_DIR.format(version, checkpoint, light_direction, scene, axis)
    light = np.load(os.path.join(light_dir, filename))
    sh = torch.tensor(light)
    #sh = sh.reshape(9, 3)
    img = drawSphere(sh, 256, is_back=True, white_bg=True)
    img = skimage.img_as_ubyte(img.permute(1, 2, 0).cpu().numpy())
    skimage.io.imsave(os.path.join(ball_dir, filename.replace("_light.npy",".png")), img)

    
    # map_with_bar, map_centered, map_clean = drawMap(sh, 256)
    # map_clean = skimage.img_as_ubyte(map_clean.permute(1, 2, 0).cpu().numpy())
    # skimage.io.imsave(os.path.join(envmap_dir, filename.replace("_light.npy",".png")), map_clean)


@notify
def main():
    jobs = []
    for version in VERSIONS:
        for checkpoint in CHECKPOINTS:
            for light_direction in [IMAGE_TYPE[0]]:
                current_dir = ROOT_DIR.format(version, checkpoint, light_direction)
                scenes = sorted(os.listdir(current_dir))
                for scene in scenes:
                    for axis in ['x','y','z']:
                        if os.path.exists(os.path.join(current_dir, scene, f"{axis}_light")):
                            files = sorted(os.listdir(os.path.join(current_dir, scene, f"{axis}_light")))
                            for filename in files:
                                if filename.endswith(".npy"):
                                    data = (version, checkpoint, light_direction, scene, axis, filename)
                                    print(data)
                                    jobs.append(data)
                                    
    with Pool(32) as p:
        r = list(tqdm(p.imap(process_file, jobs), total=len(jobs)))

if __name__ == "__main__":
    main()    

