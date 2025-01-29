import os 
from sh_utils import unfold_sh_coeff, get_ideal_normal_ball_z_up, cartesian_to_spherical, sample_from_sh
import numpy as np
from tqdm.auto import tqdm
from tonemapper import TonemapHDR
import argparse
import skimage

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')
args = parser.parse_args()


def main():
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
    coeff_dir = "shcoeffs_order100_hdr_v2"
    image_dir = "images"
    output_dir = "output/chromeball_order100_v2"
    ORDER = 100

    tonemapper = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    normal_map, _ = get_ideal_normal_ball_z_up(256) # x-forward, y-right, z-up
    theta, phi = cartesian_to_spherical(normal_map)

    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))
    
    print("CREATING QUEUES...")
    queues  = []
    for scene in scenes[:1]:
        for idx in range(25):
            queues.append((scene,idx))


    pbar = tqdm(queues[args.index::args.total])
    pbar.set_description(f"")

    for info in pbar:
        idx = info[1]
        scene = info[0]
        shading_output_dir = os.path.join(output_dir,scene)
        os.makedirs(shading_output_dir, exist_ok=True)
        os.chmod(shading_output_dir, 0o777)

        output_path = os.path.join(shading_output_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(output_path):
           continue

        shcoeff = np.load(f"{root_dir}/{coeff_dir}/{scene}/dir_{idx}_mip2.npy") # shcoeff shape (3,9)
        shcoeff = unfold_sh_coeff(shcoeff,max_sh_level=ORDER)
        shading = sample_from_sh(shcoeff, lmax=ORDER, theta=theta, phi=phi)
        shading = np.float32(shading)
        shading, _, _ = tonemapper(shading) # tonemap

        shading = np.clip(shading, 0, 1)
        shading = skimage.img_as_ubyte(shading)
        
        skimage.io.imsave(output_path,shading)
        os.chmod(output_path, 0o777)




if __name__ == "__main__":
    main()