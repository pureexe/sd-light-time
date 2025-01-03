print("IMPORIING OS")
import os 
print("IMPORTING PIL")
from PIL import Image
print("IMPORTING tqdm")
from tqdm.auto import tqdm
print("IMPORTING numpy")
import numpy as np
print("IMPORTING torch")
import torch
import torch.nn.functional as F
print("IMPORTING pyshtools")
import pyshtools
print("IMPORTING EINOPS")
from einops import rearrange
print("IMPORTING NormalBAE")
from controlnet_aux import NormalBaeDetector
print("IMPORTING CONTROLNET_UTIL")
from controlnet_aux.util import HWC3, resize_image
print("IMPORTING torchvision")
import torchvision
print("IMPORTING SKIMAGE")
import skimage


ROOT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
IMAGE_DIR = "images"
COEFF_DIR = "shcoeffs"

def sample_envmap_from_sh(shcoeff, lmax, theta, phi):
    """
    Sample envmap from sh 
    """
    assert shcoeff.shape[0] == 3 # make sure that it a 3 channel input
    output = []
    for ch in (range(3)):
        coeffs = pyshtools.SHCoeffs.from_array(shcoeff[ch], lmax=lmax, normalization='4pi', csphase=1)
        image = coeffs.expand(grid="GLQ", lat=theta, lon=phi, lmax_calc=lmax, degrees=False)
        output.append(image[...,None])
    output = np.concatenate(output, axis=-1)
    return output


def cartesian_to_spherical_grid(xyz_grid):
    """
    Convert a grid of Cartesian coordinates to spherical coordinates.
    
    Parameters:
        xyz_grid (ndarray): Array of shape [H, W, 3] representing Cartesian coordinates.
    
    Returns:
        tuple: Arrays (theta, phi) of shape [H, W], where
               theta is the colatitude (0 to pi),
               phi is the longitude (-pi to pi).
    """
    x, y, z = xyz_grid[..., 0], xyz_grid[..., 1], xyz_grid[..., 2]
    r = np.linalg.norm(xyz_grid, axis=-1)
    theta = np.arccos(np.clip(z / r, -1, 1))  # Colatitude
    phi = np.arctan2(y, x)  # Longitude
    return theta, phi


def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                sh_coeff[i, 1, j, k] = flatted_coeff[i, c]
                c +=1
            for k in range(j+1):
                sh_coeff[i, 0, j, k] = flatted_coeff[i, c]
                c += 1
    return sh_coeff

def remap_theta_phi(theta, phi):
    """
    Remap theta from  (0 to pi) to (-pi/2 to pi/2)
    and from (-pi to pi) to (0 to 2 pi)
    """
    theta = theta - np.pi / 2 
    phi = phi + np.pi
    return theta, phi

def get_queues():
    #scenes = sorted(os.listdir(os.path.join(ROOT_DIR, IMAGE_DIR)))
    scenes = ['14n_copyroom1', '14n_copyroom10','14n_copyroom8']
    #scenes = ['14n_copyroom8']
    queues  = []
    for scene in scenes:
        for idx in range(25):
            queues.append((scene,idx))
    return queues

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
            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()

        return normal

def main():
    print("GETTING QUEUE")
    queues = get_queues()
    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor = preprocessor.to('cuda')
    
    #queues = queues[:25]
    pbar = tqdm(queues)
    pbar.set_description(f"")
    for info in pbar:
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")
        idx = info[1]
        scene = info[0]
        image = Image.open(f"{ROOT_DIR}/{IMAGE_DIR}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        normal_map = preprocessor(image, output_type="pt") 
        normal_map = torch.from_numpy(normal_map)
        normal_map = normal_map.permute(2,0,1)[None]
        # normal map shape [1,3,512,512] range (-1,1)
        # load shcoeff 
        shcoeff = np.load(f"{ROOT_DIR}/{COEFF_DIR}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        shcoeff = unfold_sh_coeff(shcoeff) #shape [3,2,3,3]

        theta, phi = cartesian_to_spherical_grid(normal_map[0].permute(1,2,0).numpy())
        #theta, phi = remap_theta_phi(theta, phi)
        shading = sample_envmap_from_sh(shcoeff, 2, theta, phi)
        shading = np.clip(shading, 0.0, 1.0)
        
        os.makedirs(os.path.join("output", "ldr_expand_no_remap", scene),exist_ok=True)
        output_path = os.path.join("output", "ldr_expand_no_remap",  scene, f"dir_{idx}_mip2.png")
        #image.save(output_path)
        skimage.io.imsave(output_path, skimage.img_as_ubyte(shading))

if __name__ == "__main__":
    main()