PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/shcoeffs_order100_hdr/14n_copyroom1/dir_5_mip2.npy"

import pyshtools
import numpy as np 
import os
import ezexr 
from tqdm.auto import tqdm 

ORDER = 2

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

def compute_background(
        sh, image_width=512, lmax=2
    ):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

    
        theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
        phi = np.linspace(0, np.pi * 2, 2*image_width)
    
        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    return output_image


def main_old():
    shcoeff = np.load(PATH)
    shcoeff = unfold_sh_coeff(shcoeff, max_sh_level=ORDER)
    print("UNFOLDED SH COEFFICIENTS computed")
    # compute background
    background = compute_background(shcoeff, lmax=ORDER)
    ezexr.imwrite("background_hdr_dir_5_mip2.exr", background)

def main():
    DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/shcoeffs_order100_hdr/14n_copyroom1"
    OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250111_create_mesh_from_depth_and_focal/output/precompute_copyroom01"
    files = sorted(os.listdir(DIR))
    files = [f for f in files if f.endswith('npy')]
    for filename in tqdm(files):
        shcoeff = np.load(os.path.join(DIR,filename))
        shcoeff = unfold_sh_coeff(shcoeff, max_sh_level=ORDER)
        # compute background
        background = compute_background(shcoeff, lmax=ORDER)
        ezexr.imwrite(os.path.join(OUTPUT_DIR,filename.replace('.npy','.exr')), background)



if __name__ == "__main__":
    main()

