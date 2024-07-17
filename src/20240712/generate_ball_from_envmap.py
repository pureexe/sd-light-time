from LineNotify import notify 
from ezexr import imread 
import os 
import numpy as np
import skimage
from multiprocessing import Pool
from tqdm.auto import tqdm

EXR_DIR = "datasets/face/face2000_single/exr"
EV0_DIR = "datasets/face/face2000_single/ev0"
OUTPUT_DIR = "datasets/face/face2000_single_viz/viz_envmap"

def process(meta):
    filename = meta['filename']
    subdir = meta['subdir']
    image_dir = os.path.join(EXR_DIR, subdir)
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir,exist_ok=True)
    envmap = imread(os.path.join(image_dir, filename + ".exr"))
    ev0_map = skimage.io.imread(os.path.join(EV0_DIR, subdir, filename + ".png"))
    ev0_map = skimage.img_as_float(ev0_map)
    envmap_gray = 0.299*envmap[..., 0]  + 0.587*envmap[..., 1] + 0.114*envmap[..., 2]
    highest_position = np.unravel_index(np.argmax(envmap_gray), envmap_gray.shape)
    highest_value = envmap_gray[highest_position]
    mask = np.abs(envmap_gray - highest_value) < 1e-3
    
    ev0_map[mask, 0] = 1
    ev0_map[mask, 1] = 0
    ev0_map[mask, 2] = 0

    ev0_map = skimage.img_as_ubyte(ev0_map)
    out_img_path = os.path.join(out_dir, filename + ".png")
    skimage.io.imsave(out_img_path, ev0_map)
    


#@notify
def main():
    metas = []
    for subdir in os.listdir(EXR_DIR):
        for filename in os.listdir(os.path.join(EXR_DIR, subdir)):
            if filename.endswith(".exr"):
                filename = filename.split(".")[0]
                metas.append({
                    "filename": filename,
                    "subdir": subdir
                })
    with Pool(24) as p:
      r = list(tqdm(p.imap(process, metas), total=len(metas))) 
    

if __name__ == "__main__":
    main()