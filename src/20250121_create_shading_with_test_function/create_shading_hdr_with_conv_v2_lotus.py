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
print("IMPORTING THE NormalBAE")
from controlnet_aux import NormalBaeDetector
print("IMPORTING PYSHTOOLS")
import pyshtools
print("IMPORTING MULTIPROCESSING")
from multiprocessing import Pool
from functools import partial
print("IMPORTING CONTROLNET_UTIL")
from controlnet_aux.util import HWC3, resize_image
print("IMPOORING EINOPS")
from einops import rearrange
from tonemapper import TonemapHDR
import ezexr
print("IMPORT DONE")
import cv2
import argparse

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')

args = parser.parse_args()


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

def cartesian_to_spherical(vectors):
    """
    Converts unit vectors to spherical coordinates (theta, phi).

    Parameters:
    vectors (numpy.ndarray): Input array of shape (..., 3), representing unit vectors.

    Returns:
    tuple: A tuple containing two arrays:
        - theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
        - phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].
    """
    # Ensure input is a numpy array
    vectors = np.asarray(vectors)

    # Validate shape
    if vectors.shape[-1] != 3:
        raise ValueError("Input must have shape (..., 3).")

    # Extract components of the vectors
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]

    # Calculate theta (latitude angle)
    theta = np.arcsin(y)  # arcsin gives range [-pi/2, pi/2]

    # Calculate phi (longitude angle)
    phi = np.arctan2(x, z)  # atan2 accounts for correct quadrant
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # Normalize phi to range [0, 2*pi]

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

def apply_integrate_conv(shcoeff):
    # apply integrate on diffuse surface 
    # @see https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2
    A = np.array([
        np.pi, # 0
        2*np.pi / 3, # 1
        np.pi / 4, # 2
    ])
    for j in range(3):
        # check if it still access
        if j < shcoeff.shape[2]:
            shcoeff[:,:,j] = A[j] * shcoeff[:,:,j]
    return shcoeff


def filmic_tone_map(rgb_image, gamma=2.4):
    """
    Applies a Filmic tone mapping curve to an RGB image in linear space,
    followed by gamma correction for display.

    Parameters:
        rgb_image (numpy.ndarray): Input HDR image in linear RGB space,
                                   shape (H, W, 3), values may exceed 1.
        gamma (float): Gamma value for gamma correction (default is 2.4).

    Returns:
        numpy.ndarray: Tone-mapped and gamma-corrected RGB image with values [0, 1].
    """
    # Constants for the Filmic curve (approximation of Blender's Filmic)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    # Ensure input is a NumPy array
    rgb_image = np.array(rgb_image, dtype=np.float32)

    # Apply Filmic tone mapping curve
    tone_mapped = (rgb_image * (a * rgb_image + b)) / (rgb_image * (c * rgb_image + d) + e)

    # Apply gamma correction to map to display-referred space
    gamma_corrected = np.power(tone_mapped, 1.0 / gamma)

    # Clamp values to [0, 1] range after gamma correction
    output_image = np.clip(gamma_corrected, 0, 1)

    return output_image


def get_lotus_normal(idx):
    normal_path = f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/normal_lotus/dir_{idx}_mip2.npz"
    normal_map = np.load(normal_path)
    normal_map = normal_map[normal_map.files[0]]
    return normal_map


def main():


    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
    image_dir = "images"
    coeff_dir = "shcoeffs_order100_hdr"
    output_dir = "control_shading_from_hdr27coeff_conv_v3"
    mode = 'bae'
    ORDER = 2

    print("LOADING PREPROCESSOR")
    os.makedirs(os.path.join(root_dir, output_dir), exist_ok=True)
    os.chmod(os.path.join(root_dir, output_dir), 0o777)

    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor.to('cuda')

    print("CREATING QUEUES...")
    queues  = []
    for scene in scenes[:1]:
        for idx in range(25):
            queues.append((scene,idx))
            
    tonemapper = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

    pbar = tqdm(queues[args.index::args.total])
    pbar.set_description(f"")
    for info in pbar:
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")

        idx = info[1]
        scene = info[0]
        #shading_output_dir = os.path.join(root_dir,output_dir,scene)
        shading_output_dir = "output/shadings_v2_lotus_persepctive_div4_order100_gt/"
        output_path = os.path.join(shading_output_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(output_path):
           continue
        #image = Image.open(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
        #normal_map = preprocessor(image, output_type="pt")
        normal_map = get_lotus_normal(idx)
        
        theta, phi = cartesian_to_spherical(normal_map)


        #shcoeff = np.load(f"{root_dir}/{coeff_dir}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        #shcoeff = np.load(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/shcoeff_perspective_order100_gt/dir_{idx}_mip2.npy")
        #shcoeff = np.load(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/shcoeff_orthographic_order100_gt/dir_{idx}_mip2.npy")
        shcoeff = np.load(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/scene_inspect/14n_copyroom1/000000/shcoeff_perspective_div4_order100_gt/dir_{idx}_mip2.npy")
        
        shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=ORDER)

        shcoeff = apply_integrate_conv(shcoeff)

        shading = sample_envmap_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
        
        os.makedirs(shading_output_dir, exist_ok=True)
        os.chmod(shading_output_dir, 0o777)

        shading = np.float32(shading)
        shading, _, _ = tonemapper(shading) # tonemap
        
        shading = np.clip(shading, 0, 1)
        shading = skimage.img_as_ubyte(shading)
        
        skimage.io.imsave(output_path,shading)
        os.chmod(output_path, 0o777)






if __name__ == "__main__":
    main()