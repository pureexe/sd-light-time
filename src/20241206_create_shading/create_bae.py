import os
from PIL import Image
from transformers import pipeline
from tqdm.auto import tqdm
import warnings
import torch
import skimage
import numpy as np
import cv2
from controlnet_aux import NormalBaeDetector
from multiprocessing import Pool
from functools import partial
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from controlnet_aux.processor import Processor
from controlnet_aux.util import HWC3, resize_image
from einops import rearrange

pi = np.pi
CONSTANT_FACTOR = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()

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

class NormalBaeDetectorPT(NormalBaeDetector):
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            #normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            #normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return normal

def main():
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test"
    image_dir = "images"
    coeff_dir = "shcoeffs"
    output_dir = "control_shading_from_ldr27coeff"
    mode = 'bae'
    

    os.makedirs(os.path.join(root_dir, output_dir), exist_ok=True)
    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor.to('cuda')
    
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
        image = Image.open(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        normal_map = preprocessor(image, output_type="pt")
        normal_map = torch.from_numpy(normal_map)
        normal_map = normal_map.permute(2,0,1)[None]
        shcoeff = np.load(f"{root_dir}/{coeff_dir}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        shcoeff = torch.from_numpy(shcoeff).permute(1,0)[None] # shcoeff [BATCH, 9, 3]
        shading = add_SHlight(normal_map, shcoeff)
        shading = shading[0].permute(1,2,0)
        shading = skimage.img_as_ubyte(shading)
        shading_output_dir = os.path.join(root_dir,output_dir,scene)
        os.makedirs(shading_output_dir, exist_ok=True)
        
        skimage.io.imsave(os.path.join(shading_output_dir, f"dir_{idx}_mip2.png"),shading)




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