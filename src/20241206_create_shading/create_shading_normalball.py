print("IMPORTING OS")
import os
from PIL import Image
print("IMPORTING PIPELINE")
from transformers import pipeline
from tqdm.auto import tqdm
import warnings
print("IMPORTING TORCH")
import torch
print("IMPORTING SKIMAGE")
import skimage
print("IMPORTING NUMPY")
import numpy as np
print("IMPORTING CV2")
import cv2
print("IMPORTING MULTIPROCESSING")
from multiprocessing import Pool
from functools import partial
print("IMPORTING CONTROLNET_UTIL")
from controlnet_aux.util import HWC3, resize_image
print("IMPOORING EINOPS")
from einops import rearrange
from tonemapper import TonemapHDR
print("IMPORT DONE")

pi = np.pi
CONSTANT_FACTOR = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def add_SHlight(normal_images, sh_coeff):
    '''
        sh_coeff: [bz, 9, 3]
    '''
    N = normal_images
    sh = torch.stack([
            N[:,0]*0.+1., N[:,0], N[:,1], \
            N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
            N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
            ], 
            1) # [bz, 9, h, w]
    sh = sh*CONSTANT_FACTOR[None,:,None,None]
    shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
    return shading

def main():

    normal_image, _ = get_ideal_normal_ball(128)
    normal_image_flip = normal_image.copy()
    normal_image_flip[:,:,2] = -normal_image_flip[:,:,2]
    normal_image = np.concatenate([normal_image,normal_image_flip],axis=1)
    
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
    image_dir = "images"
    # coeff_dir = "shcoeffs_order2_hdr"
    # output_dir = "normalball_from_hdr27coeff"
    coeff_dir = "shcoeffs"
    output_dir = "normalball_from_ldr27coeff"
    mode = 'bae'
    
    print("LOADING PREPROCESSOR")
    os.makedirs(os.path.join(root_dir, output_dir), exist_ok=True)
    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    tonemapper = TonemapHDR(gamma=2.4, percentile=90, max_mapping=0.9)
    print("QUUINING...")
    queues  = []
    for scene in scenes:
        for idx in range(25):
            queues.append((scene,idx))
    
    pbar = tqdm(queues)
    pbar.set_description(f"")
    for info in pbar:
        
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")

        idx = info[1]
        scene = info[0]
        shading_output_dir = os.path.join(root_dir,output_dir,scene)
        output_path = os.path.join(shading_output_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(output_path):
            continue
        image = Image.open(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        normal_map = normal_image
        normal_map = torch.from_numpy(normal_map)
        normal_map = normal_map.permute(2,0,1)[None]
        shcoeff = np.load(f"{root_dir}/{coeff_dir}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        shcoeff = torch.from_numpy(shcoeff).permute(1,0)[None] # shcoeff [BATCH, 9, 3]
        shading = add_SHlight(normal_map, shcoeff)
        shading = shading[0].permute(1,2,0).numpy()
        shading, _, _ = tonemapper(shading)
        shading = np.clip(shading, 0, 1)
        shading = skimage.img_as_ubyte(shading)
        os.makedirs(shading_output_dir, exist_ok=True)
        
        skimage.io.imsave(output_path,shading)




    # for idx in range(25):
    #     image = Image.open(f"/images/14n_copyroom1/dir_{idx}_mip2.jpg").convert("RGB")
    #     normal_map = preprocessor(image, output_type="pt")
    #     normal_map = torch.from_numpy(normal_map)
    #     normal_map = normal_map.permute(2,0,1)[None]

    #     shcoeff = np.load(f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/shcoeffs/14n_copyroom1/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        
    #     shcoeff = torch.from_numpy(shcoeff).permute(1,0)[None] # shcoeff [BATCH, 9, 3]

    #     shading = add_SHlight(normal_map, shcoeff)
        
    #     print(f"MIN {shading.min()}, MAX {shading.max()}")

    #     shading = shading[0].permute(1,2,0)
    #     shading = skimage.img_as_ubyte(shading)
    #     skimage.io.imsave(f"output/shading_{idx}.png",shading)

    # print("DONE!")
    # print(shading.min())
    # print(shading.max())
    # exit()


    # print(shcoeff.shape)

    # shading = None
    # #np.save("normal_bae.npy", normal_map)


if __name__ == "__main__":
    main()