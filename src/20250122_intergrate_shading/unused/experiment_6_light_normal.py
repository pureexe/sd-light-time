from sh_utils  import get_shcoeff, apply_integrate_conv, get_ideal_normal_ball_z_up, cartesian_to_spherical, sample_from_sh, from_x_left_to_z_up
import skimage
from tqdm.auto import tqdm
from tonemapper import TonemapHDR
import numpy as np
import os
from PIL import Image
from efficient_sh import NormalBaeDetectorPT

def main():
    axis_names = [
        "x_plus",
        "x_minus",
        "y_plus",
        "y_minus",
        "z_plus",
        "z_minus"
    ]

    preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
    preprocessor.to('cuda')

    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
    idx = 0
    scene = "14n_copyroom1"
    image_dir = "images"
    image = Image.open(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg").convert("RGB")
    normal_map = preprocessor(image, output_type="pt")
    #normal_map = from_x_left_to_z_up(normal_map)



    #normal_map, mask = get_ideal_normal_ball_z_up(256)
    theta, phi = cartesian_to_spherical(normal_map) 
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

    for axis_name in axis_names:
        image = skimage.io.imread(f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250122_intergrate_shading/data/axis_light/{axis_name}.png")
        image = np.concatenate([image[...,None],image[...,None],image[...,None]], axis=-1)
        image = skimage.img_as_ubyte(image)
        sh_coeff = get_shcoeff(image, Lmax=2)
        integrated_shcoeff = apply_integrate_conv(sh_coeff)
        shading = sample_from_sh(integrated_shcoeff, lmax=2, theta=theta, phi=phi)
        #tonemapped_shading = np.clip(shading, 0, 1)
        tonemapped_shading, _, _ = tonemap(shading)
        skimage.io.imsave(os.path.join("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250122_intergrate_shading/output/axis_light_normal_without_x_left_to_z_up", f"{axis_name}.png"), skimage.img_as_ubyte(tonemapped_shading))



if __name__ == "__main__":
    main()