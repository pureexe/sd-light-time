from sh_utils import get_shcoeff, get_ideal_normal_ball_z_up, cartesian_to_spherical, sample_from_sh
from tonemapper import TonemapHDR
import skimage
import os

def get_envmap_from_file():
    """
    Get environemnt map image
    Returns: 
    - np.ndarray: image in HWC format in range [0,1]
    """
    image = skimage.io.imread("data/lorem_ipsum2.png")[...,:3]
    image = skimage.img_as_float(image)
    return image

def main():
    ORDER = 100
    OUTPUT_DIR = "output/ball_from_envmap"

    image = get_envmap_from_file()
    shcoeff = get_shcoeff(image, Lmax=ORDER)
    normal_map, mask = get_ideal_normal_ball_z_up(512)

    theta, phi = cartesian_to_spherical(normal_map) 

    shading = sample_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    tonemapped_shading, _, _ = tonemap(shading)
    skimage.io.imsave(os.path.join(OUTPUT_DIR, "ball_from_envmap.png"), skimage.img_as_ubyte(tonemapped_shading))
    print("DONE")



if __name__ == "__main__":
    main()