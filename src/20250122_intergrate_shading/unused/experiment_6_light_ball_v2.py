from sh_utils  import get_shcoeff, apply_integrate_conv, get_ideal_normal_ball_z_up, cartesian_to_spherical, sample_from_sh
import skimage
from tqdm.auto import tqdm
from tonemapper import TonemapHDR
import numpy as np
import ezexr
import os


def main():
    axis_names = [
        "x_plus",
        "x_minus",
        "y_plus",
        "y_minus",
        "z_plus",
        "z_minus"
    ]
    normal_map, mask = get_ideal_normal_ball_z_up(257)
    theta, phi = cartesian_to_spherical(normal_map) 
    skimage.io.imsave("theta_257.png", skimage.img_as_ubyte(((theta / (np.pi / 2)) + 1.0) / 2.0)) # pi/2, -pi/2
    skimage.io.imsave("phi_257.png", skimage.img_as_ubyte((phi / (np.pi * 2))))
    print("DONE")
    exit()
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)

    for axis_name in axis_names:
        print("processing... ", axis_name)

        image = skimage.io.imread(f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250122_intergrate_shading/data/axis_light/{axis_name}.png")
        image = np.concatenate([image[...,None],image[...,None],image[...,None]], axis=-1)
        image = skimage.img_as_ubyte(image)
        sh_coeff = get_shcoeff(image, Lmax=2)
        integrated_shcoeff = apply_integrate_conv(sh_coeff)
        shading = sample_from_sh(integrated_shcoeff, lmax=2, theta=theta, phi=phi)
        #tonemapped_shading = np.clip(shading, 0, 1)
        #tonemapped_shading, _, _ = tonemap(shading)
        print("saving... ", axis_name)
        ezexr.imwrite(os.path.join("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250122_intergrate_shading/output/axis_light_ball", f"{axis_name}.exr"), shading)
        #skimage.io.imsave(os.path.join("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250122_intergrate_shading/output/axis_light_ball", f"{axis_name}.png"), skimage.img_as_ubyte(tonemapped_shading))



if __name__ == "__main__":
    main()