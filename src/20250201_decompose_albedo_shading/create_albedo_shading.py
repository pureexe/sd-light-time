from tonemapper import TonemapHDR
import argparse 
from chrislib.data_util import load_image
from intrinsic.pipeline import load_models, run_pipeline
import skimage
import os
from tqdm.auto import tqdm 
import numpy as np 

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')

args = parser.parse_args()


def main():
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
    image_dir = "images"
    diffuse_dir = "control_intrinsic_shading_diffuse"
    albedo_dir = "control_intrinsic_albedo_v2"
    mode = 'bae'
    ORDER = 2
    print("BEFORE LOAD MODEL")
    models = load_models('v2')
 
    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    output_albedo_dir = os.path.join(root_dir,albedo_dir)
    output_diffuse_dir = os.path.join(root_dir,diffuse_dir)
    os.makedirs(output_albedo_dir, exist_ok=True)
    os.chmod(output_albedo_dir, 0o777)
    os.makedirs(output_diffuse_dir, exist_ok=True)
    os.chmod(output_diffuse_dir, 0o777)
 
    print("CREATING QUEUES...")
    queues  = []
    for scene in scenes[:1]:
        for idx in range(25):
            queues.append((scene,idx))

    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    pbar = tqdm(queues[args.index::args.total])
    pbar.set_description(f"")

    for info in pbar:
        
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")

        idx = info[1]
        scene = info[0]
        output_albedo_dir = os.path.join(root_dir,albedo_dir,scene)
        output_diffuse_dir = os.path.join(root_dir,diffuse_dir,scene)
        os.makedirs(output_albedo_dir, exist_ok=True)
        os.chmod(output_albedo_dir, 0o777)
        os.makedirs(output_diffuse_dir, exist_ok=True)
        os.chmod(output_diffuse_dir, 0o777)
        albedo_path = os.path.join(output_albedo_dir, f"dir_{idx}_mip2.png")
        diffuse_path = os.path.join(output_diffuse_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(albedo_path) and os.path.exists(diffuse_path):
           continue

        image = load_image(f"{root_dir}/{image_dir}/{scene}/dir_{idx}_mip2.jpg")

        results = run_pipeline(models, image)

        albedo = results['hr_alb']
        albedo = skimage.transform.resize(albedo, (512, 512), anti_aliasing=True)

        albedo = np.clip(albedo, 0, 1)
        albedo = skimage.img_as_ubyte(albedo)
        skimage.io.imsave(albedo_path,albedo)
        os.chmod(albedo_path, 0o777)

        diffuse_shading = results['dif_shd']
        diffuse_shading = skimage.transform.resize(diffuse_shading, (512, 512), anti_aliasing=True)
        diffuse_shading_mapped, _, _ = tonemap(diffuse_shading)
        diffuse_shading_mapped = np.clip(diffuse_shading_mapped, 0, 1)
        diffuse_shading_mapped = skimage.img_as_ubyte(diffuse_shading_mapped)
        skimage.io.imsave(diffuse_path,diffuse_shading_mapped)
        os.chmod(diffuse_path, 0o777)



if __name__ == "__main__":
    main()