import skimage
import numpy as np
from sh_utils import get_shcoeff, get_ideal_normal_ball, compute_background
from create_shading_hdr_with_conv import cartesian_to_spherical, apply_integrate_conv, sample_envmap_from_sh
from tonemapper import TonemapHDR

ORDER = 100

def main():
    tonemapper = TonemapHDR(gamma=2.4, percentile=90, max_mapping=0.9)

    print("Getting spherical harmonic environment map")
    coordinate = skimage.io.imread("coordinates.png")[...,:3]
    coordinate = skimage.img_as_float(coordinate)
    shcoeff = get_shcoeff(coordinate,Lmax=ORDER)
    background = compute_background(shcoeff, lmax=ORDER)
    skimage.io.imsave("background.png", skimage.img_as_ubyte(np.clip(background, 0.0, 1.0)))
    normal_map, mask = get_ideal_normal_ball(256) # tested in diffusion light
    # save normal ball for future visualize sation
    normal_rgb = (normal_map + 1.0) / 2.0 
    skimage.io.imsave("normal_ball.png", skimage.img_as_ubyte(normal_rgb))

    theta, phi = cartesian_to_spherical(normal_map)

    shading_before = sample_envmap_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
    
    shading_before, _, _ = tonemapper(shading_before) # tonemap
    
    skimage.io.imsave("shading_before.png", skimage.img_as_ubyte(np.clip(shading_before, 0.0, 1.0)))

    shcoeff = apply_integrate_conv(shcoeff)

    shading = sample_envmap_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)


    shading, _, _ = tonemapper(shading) # tonemap
    
    shading = np.clip(shading, 0, 1)
    shading = skimage.img_as_ubyte(shading)

    skimage.io.imsave("shading_after.png",shading)




    print("✅✅✅ All test done ✅✅✅")

if __name__ == "__main__":
    main()