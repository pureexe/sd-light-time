import os 
import numpy as np 
from LineNotify import notify
from sphere_helper import drawSphere
import skimage
from PIL import Image
from multiprocessing import Pool 
from tqdm.auto import tqdm
import torch

INPUT_DIR = "datasets/face/face2000_single/light"
OUTPUT_DIR = "datasets/face/face2000_single/ball_npy"

def process(meta):
    filename = meta['filename']
    subdir = meta['subdir']
    image_dir = os.path.join(INPUT_DIR, subdir)
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir,exist_ok=True)

    sh = np.load(os.path.join(image_dir, filename))
    sh = torch.tensor(sh)
    
    # save front_image
    image = drawSphere(sh, img_size=256, is_back=False)
    image = image.permute(1, 2, 0).numpy()
    #image = skimage.img_as_ubyte(image)
    #img = Image.fromarray(image)
    #img.save(os.path.join(out_dir, filename.replace("_light.npy", "_front.png")))
    np.save(os.path.join(out_dir, filename.replace("_light.npy", "_front.npy")), image)

    # save back_image
    image = drawSphere(sh, img_size=256, is_back=True)
    image = image.permute(1, 2, 0).numpy()
    # image = skimage.img_as_ubyte(image)
    # img = Image.fromarray(image)
    # img.save(os.path.join(out_dir, filename.replace("_light.npy", "_rear.png")))
    np.save(os.path.join(out_dir, filename.replace("_light.npy", "_back.npy")), image)



@notify
def main():
    
    # read all file in path
    metas = []
    for subdir in sorted(os.listdir(INPUT_DIR)):
        image_dir = os.path.join(INPUT_DIR, subdir)
        for filename in sorted(os.listdir(image_dir)):
            metas.append({
                'subdir': subdir,
                'filename': filename
            })

    with Pool(24) as p:
      r = list(tqdm(p.imap(process, metas), total=len(metas))) 
            
            




if __name__ == "__main__":
    main()