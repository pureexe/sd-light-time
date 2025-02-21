import os 
from tqdm.auto import tqdm 
import diffusers
import numpy as np 
import ezexr
import skimage
import torch 

SPLIT = "train"
PREDICT_DIR = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/{SPLIT}"
SOURCE_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT}/images"

def get_basis( normal):
        """
        get the basis function for the spherical harmonics
        @see https://github.com/diffusion-face-relighting/difareli_code/blob/2dd24a024f26d659767df6ecc8da4ba47c55e7a8/guided_diffusion/models/renderer.py#L25
        """
        sh_constant = torch.tensor([
            1/np.sqrt(4*np.pi), 
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))), 
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))),
            ((2*np.pi)/3)*(np.sqrt(3/(4*np.pi))), 
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))),
            (np.pi/4)*(3)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(3/2)*(np.sqrt(5/(12*np.pi))), 
            (np.pi/4)*(1/2)*(np.sqrt(5/(4*np.pi)))]
        ).float()

        # verify that we have normal shape [B,3,H,W]
        assert len(normal.shape) == 4 and normal.shape[1] == 3
        
        basis = torch.stack([
            normal[:,0]*0.+1.,                  # 1
            normal[:,0],                        # X
            normal[:,1],                        # Y
            normal[:,2],                        # Z
            normal[:,0] * normal[:,1],          # X*Y
            normal[:,0] * normal[:,2],          # X*Z
            normal[:,1] * normal[:,2],          # Y*Z
            normal[:,0]**2 - normal[:,1]**2,    # X**2 - Y**2
            3*(normal[:,2]**2) - 1,             # 3(Z**2) - 1
            ], 
            axis=1
        ) # [bz, 9, h, w]

        sh_constant = sh_constant[None, :, None, None].to(normal.device)
        
        basis = basis * sh_constant # [bz, 9, h, w]

        # verify that we use order 2 which has 9 basis 
        assert basis.shape[1] == 9
        return basis

def render_image(shcoeffs, normal, albedo = None):
        """
        render image from normal using spherical harmonics and albedo
        O = albedo * \sum_{l,m} shcoeffs * BASIS(l,m,normal)
        """
        basis = get_basis(normal) # [bz, 9, h, w]
        shading = torch.sum(
            shcoeffs[:, :, :, None, None] # [bz, 3, 9, 1, 1]
            * basis[:, None, :, :, :], # [bz, None, 9, h, w]
            axis=2
        ) # [bz, 3, h, w]  

        if albedo is not None:
            # albedo range [0,1] * shading range [0,1] to image range [0,1]
            rendered = albedo * shading
        else:
            rendered = shading

        assert rendered.shape[1:] == normal.shape[1:] and shcoeffs.shape[0] == rendered.shape[0] # [bz, 3, h, w]
        return rendered

def get_avg_images(path):
    files = sorted(os.listdir(path))
    images = []
    for f in files:
        try:
            image_path = os.path.join(path,f)
            img = skimage.io.imread(image_path)
        except:
            continue
        img = skimage.img_as_float(img)
        images.append(img[None])    

    images = np.concatenate(images,axis=0)
    images = images.mean(axis=0)
    images = torch.tensor(images).permute(2,0,1)
    return images

def get_shcoeff(scene):
    # get lastest version of scnees
    scene_root = os.path.join(PREDICT_DIR, scene, 'lightning_logs')
    versions = sorted(os.listdir(scene_root))
    shcoeff = np.load(os.path.join(scene_root, versions[-1], 'shcoeffs.npy'))
    return shcoeff

def log_normalize(image):
    log_image = np.log1p(image)  # log(1 + I) to prevent log(0) issues
    return log_image / np.max(log_image)  # Normalize by new max value

@torch.inference_mode()
def main():
    scenes = sorted(os.listdir(PREDICT_DIR))
    for scene in tqdm(scenes):
        scene_root = os.path.join(PREDICT_DIR, scene, 'lightning_logs')
        versions = sorted(os.listdir(scene_root))
        out_dir = os.path.join(scene_root, versions[-1], 'shading_log1p') # from neural grapffer
        if os.path.exists(out_dir):
            continue
        exr_dir = os.path.join(scene_root, versions[-1], 'shading_exr')
        os.makedirs(out_dir, exist_ok=True)       
        for image_id in tqdm(range(25)):
            image = ezexr.imread(os.path.join(exr_dir, f'dir_{image_id}_mip2.exr'))
            norm_img = log_normalize(image)
            norm_img = skimage.img_as_ubyte(norm_img)
            out_path = os.path.join(out_dir, f'dir_{image_id}_mip2.png')
            skimage.io.imsave(out_path, norm_img)


if __name__ == "__main__":
    main()