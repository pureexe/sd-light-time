import os 
import numpy as np
import skimage.io 
from LineNotify import notify
from sphere_helper import drawSphere, genSurfaceNormals
import skimage
from PIL import Image
from multiprocessing import Pool 
from tqdm.auto import tqdm
import torch
import json

INPUT_DIR = "datasets/face/face2000_single/images"
NPY_DIR = "datasets/face/face2000_single/ball_npy"
OUTPUT_DIR = "datasets/face/face2000_single_viz/ball_bright_dark_back"
JSON_FILE = "datasets/face/face2000_single_viz/bright_dark_back.json"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def process(meta):
    filename = meta['filename']
    subdir = meta['subdir']
    #image_dir = os.path.join(INPUT_DIR, subdir)
    npy_dir = os.path.join(NPY_DIR, subdir)
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir,exist_ok=True)

    chromeball_map = np.load(os.path.join(npy_dir, filename+"_back.npy"))
    _, mask_inv = genSurfaceNormals(chromeball_map.shape[0])
    mask_inv = mask_inv.numpy()
    
    # make chromeball_map to gray
    gray_map = 0.299*chromeball_map[..., 0]  + 0.587*chromeball_map[..., 1] + 0.114*chromeball_map[..., 2]

    # find brightest and darkest pixel
    brightest_position = np.unravel_index(np.argmax(gray_map), gray_map.shape)
    brightest_value = gray_map[brightest_position]
    # for the darkest spot we only find the region inside the mask
    gray_map_white_outside = gray_map.copy()
    gray_map_white_outside[mask_inv] = 2.0
    darkest_position = np.unravel_index(np.argmin(gray_map_white_outside), gray_map.shape)
    darkest_value = gray_map[darkest_position]

    # find all position that have same as the brightest value
    mask_brightest = np.abs(gray_map - brightest_value) < 1e-3
    mask_darkest = np.abs(gray_map_white_outside - darkest_value) < 1e-3

    
    # find the average location of the mask
    brightest_positions = np.argwhere(mask_brightest)
    brightest_average = np.mean(brightest_positions, axis=0)
    darkest_positions = np.argwhere(mask_darkest)
    darkest_average = np.mean(darkest_positions, axis=0)

    # save location for visualization
    p_bright = np.floor(brightest_average).astype(np.int64)
    p_dark = np.floor(darkest_average).astype(np.int64)

    min_bx = max(0, p_bright[0]-3)
    max_bx = min(chromeball_map.shape[0], p_bright[0]+3)
    min_by = max(0, p_bright[1]-3)
    max_by = min(chromeball_map.shape[1], p_bright[1]+3)

    min_dx = max(0, p_dark[0]-3)
    max_dx = min(chromeball_map.shape[0], p_dark[0]+3)
    min_dy = max(0, p_dark[1]-3)
    max_dy = min(chromeball_map.shape[1], p_dark[1]+3)

    chromeball_map[min_bx:max_bx, min_by:max_by] = [1, 0, 0]
    chromeball_map[min_dx:max_dx, min_dy:max_dy] = [0, 1, 0]

    # convert chromeball map to image
    chromeball_map = skimage.img_as_ubyte(chromeball_map)
    out_img_path = os.path.join(out_dir, filename + ".png")
    skimage.io.imsave(out_img_path, chromeball_map)

    #print(p_bright, p_dark)

    return {
        'filename': filename,
        'subdir': subdir,
        'brightest': brightest_average,
        'darkest': darkest_average,
        'image_shape': chromeball_map.shape
    }

#@notify
def main():
    
    
    # read all file in path
    metas = []
    for subdir in sorted(os.listdir(INPUT_DIR)):
        image_dir = os.path.join(INPUT_DIR, subdir)
        for filename in sorted(os.listdir(image_dir)):
            metas.append({
                'subdir': subdir,
                'filename': filename.replace(".png", "")
            })

    with Pool(24) as p:
      r = list(tqdm(p.imap(process, metas), total=len(metas))) 

    with open(os.path.join(JSON_FILE), "w") as f:
        json.dump(r, f, indent=4, cls=NpEncoder)
            
            




if __name__ == "__main__":
    main()