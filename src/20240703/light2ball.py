import os

import skimage.io 
from sphere_helper import drawSphere, drawMap
from LineNotify import notify
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm
import skimage
import torch

# ENVMAP_DIR = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/envmap"
# LIGHT_DIR = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/light"
# BALL_DIR  = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/ball2"

#ENVMAP_DIR = "output/20240703/val_axis/{}/chk{}/{}/lightning_logs/version_0/envmap"
#LIGHT_DIR = "output/20240703/val_axis/{}/chk{}/{}/lightning_logs/version_0/light"
#BALL_DIR  = "output/20240703/val_axis/{}/chk{}/{}/lightning_logs/version_0/ball"
#CHECKPOINTS = [59, 79, 99]
#CHECKPOINTS = [159, 179, 199]
#LIGHT_AXIS = ['axis_x','axis_y','axis_z']
#LIGHT_AXIS = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']
#VERSIONS = ['vae5000/3e-4', 'vae5000/1e-4', 'vae5000/5e-5']
#VERSIONS = ['vae5000/1e-4', 'vae5000/5e-5']
#VERSIONS = ['vae5000/3e-4']

#output/20240703/val_axis_lightdirection/vae5000/3e-4/chk179/axis_y/lightning_logs/version_0/crop_image

# ENVMAP_DIR = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/envmap"
# LIGHT_DIR = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/light"
# BALL_DIR  = "output/20240703/val_axis_lightdirection/{}/chk{}/{}/lightning_logs/version_0/ball"
# CHECKPOINTS = [ 179]
# LIGHT_AXIS = ['axis_x','axis_y','axis_z']
# VERSIONS = ['vae5000/3e-4']

ENVMAP_DIR = "output/20240703/val_axis_maskquater/{}/chk{}/{}/lightning_logs/version_0/envmap"
LIGHT_DIR = "output/20240703/val_axis_maskquater/{}/chk{}/{}/lightning_logs/version_0/light"
BALL_DIR  = "output/20240703/val_axis_maskquater/{}/chk{}/{}/lightning_logs/version_0/ball"
CHECKPOINTS = [ 179]
LIGHT_AXIS = ['mask_topleft','mask_topright']
VERSIONS = ['vae5000/3e-4']

def process_file(data):
    version, checkpoint, light_direction, filename = data
    ball_dir = BALL_DIR.format(version, checkpoint, light_direction)
    # make directory if not exist
    os.makedirs(ball_dir, exist_ok=True)
    
    envmap_dir = ENVMAP_DIR.format(version, checkpoint, light_direction)
    os.makedirs(envmap_dir, exist_ok=True)

    light_dir = LIGHT_DIR.format(version, checkpoint, light_direction)
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
            for light_direction in LIGHT_AXIS:
                light_dir = LIGHT_DIR.format(version, checkpoint, light_direction)
                
                for filename in os.listdir(light_dir):
                    if filename.endswith(".npy"):                        
                        data = (version, checkpoint, light_direction, filename)
                        jobs.append(data)
    with Pool(32) as p:
        r = list(tqdm(p.imap(process_file, jobs), total=len(jobs)))

if __name__ == "__main__":
    main()    

