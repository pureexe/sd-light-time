import os 
import numpy as np 
from PIL import Image
from multiprocessing import Pool
import skimage.transform
from tqdm.auto import tqdm
import torch
import skimage
import json
from LineNotify import notify

IMAGES_DIR = "datasets/face/face2000_single/images"
INPUT_DIR = "datasets/face/face2000_single/ball_npy"
VISUALIZE_DIR = "datasets/face/face2000_single_viz/ball_visualize_all"
OUTPUT_PATH = "datasets/face/face2000_single_viz/higest_position.json"

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
    image_dir = os.path.join(INPUT_DIR, subdir)
    out_dir = os.path.join(VISUALIZE_DIR, subdir)
    os.makedirs(out_dir,exist_ok=True)

    front_image = np.load(os.path.join(image_dir, filename + "_front.npy"))
    back_image = np.load(os.path.join(image_dir, filename + "_back.npy"))

    all_image = np.concatenate([front_image[None], back_image[None]], axis=0)

    # convert all image from rgb to gray scale using NTSC
    all_image_gray = 0.299*all_image[..., 0]  + 0.587*all_image[..., 1] + 0.114*all_image[..., 2]

    # find the highest value in index in all image 
    highest_position = np.unravel_index(np.argmax(all_image_gray), all_image_gray.shape)
    highest_value = all_image_gray[highest_position]
    mask = np.abs(all_image_gray - highest_value) < 1e-3
    # for visualization
    all_image_viz = np.concatenate([all_image[0], all_image[1]], axis=1)
    #viz_position = [highest_position[1], highest_position[2]]
    #if highest_position[0] == 1:
    #    viz_position[1] += front_image.shape[1]
    # all_image_viz[viz_position[0]-1:viz_position[0]+1, viz_position[1]-1:viz_position[1]+1, 0] = 1
    # all_image_viz[viz_position[0]-1:viz_position[0]+1, viz_position[1]-1:viz_position[1]+1, 1] = 0
    # all_image_viz[viz_position[0]-1:viz_position[0]+1, viz_position[1]-1:viz_position[1]+1, 2] = 0
    front_mask = mask[0]
    back_mask = mask[1]
    all_image_viz[:, :front_image.shape[1], 0][front_mask] = 1
    all_image_viz[:, :front_image.shape[1], 1][front_mask] = 0
    all_image_viz[:, :front_image.shape[1], 2][front_mask] = 0
    all_image_viz[:, front_image.shape[1]:, 0][back_mask] = 1
    all_image_viz[:, front_image.shape[1]:, 1][back_mask] = 0
    all_image_viz[:, front_image.shape[1]:, 2][back_mask] = 0
    source_image = skimage.io.imread(os.path.join(IMAGES_DIR, subdir, filename + ".png"))
    source_image = skimage.transform.resize(source_image, front_image.shape[:2])
    source_image = skimage.img_as_ubyte(source_image)
    all_image_viz = skimage.img_as_ubyte(all_image_viz)
    image_with_viz = np.concatenate([source_image, all_image_viz], axis=1)
    img = Image.fromarray(image_with_viz)
    img.save(os.path.join(out_dir, filename + ".png"))
    return {
        'subdir': subdir,
        'filename': filename,
        'highest_position': highest_position,
        'highest_value': highest_value,
        'image_size': all_image_gray.shape
    }

@notify
def main():
    metas = []
    for subdir in sorted(os.listdir(IMAGES_DIR)):
        image_dir = os.path.join(IMAGES_DIR, subdir)
        for filename in sorted(os.listdir(image_dir)):
            metas.append({
                'subdir': subdir,
                'filename': filename.replace(".png","")
            })
    

    with Pool(24) as p:
        r = list(tqdm(p.imap(process, metas), total=len(metas)))
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(r, f, indent=4, cls=NpEncoder)



if __name__ == "__main__":
    main()