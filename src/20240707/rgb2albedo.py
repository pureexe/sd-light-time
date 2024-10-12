from chrislib.general import view, tile_imgs, view_scale, uninvert
from chrislib.data_util import load_image

from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models
import skimage
import os 
from tqdm.auto import tqdm
import numpy as np

intrinsic_model = load_models('paper_weights')

DATASET_PATH = "datasets/face/face2000_single"


for subdir in sorted(os.listdir(DATASET_PATH + '/images')):
    for fname in tqdm(sorted(os.listdir(os.path.join(DATASET_PATH, 'images', subdir)))):
        if fname.endswith(".png"):
            fname = fname.replace(".png","")
            img = load_image(f'datasets/face/face2000_single/images/{subdir}/{fname}.png')

            # run the image through the pipeline (use R0 resizing dicussed in the paper)
            result = run_pipeline(
                intrinsic_model,
                img,
                resize_conf=0.0,
                maintain_size=True,
                linear=False,
                device='cuda'
            )

            # convert the inverse shading to regular shading for visualization
            shd = uninvert(result['inv_shading'])
            alb = result['albedo']

            # save alb and shd to npy in intrinsic directory
            os.makedirs(DATASET_PATH+'/intrinsic', exist_ok=True)
            os.makedirs(DATASET_PATH+'/intrinsic/'+subdir, exist_ok=True)
            np.save(DATASET_PATH+'/intrinsic/'+subdir+'/'+fname+'_albedo.npy', alb)
            np.save(DATASET_PATH+'/intrinsic/'+subdir+'/'+fname+'_shading.npy', shd)

            # from linear image to sRGB
            alb = view(alb)
            alb = skimage.img_as_ubyte(alb)

            # albedo 
            os.makedirs(DATASET_PATH+'/albedo', exist_ok=True)
            os.makedirs(DATASET_PATH+'/albedo/'+subdir, exist_ok=True)

            skimage.io.imsave(DATASET_PATH+'/albedo/'+subdir+'/'+fname+'.png', alb)