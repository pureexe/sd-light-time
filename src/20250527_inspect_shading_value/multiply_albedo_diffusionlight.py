import os 
import ezexr
import skimage
import numpy as np 
from tqdm.auto import tqdm
from tonemapper import TonemapHDR
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Multiply albedo with shading and save the result.")
    parser.add_argument("--albedo_dir", type=str, default="/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_albedo_from_fitting_v2",
                        help="Directory containing albedo images.")
    parser.add_argument("--shading_dir", type=str, default="/pure/f1/datasets/multi_illumination/lstsq_image_real_shading/v0/train/shadings/",
                        help="Directory containing shading images.")
    parser.add_argument("--output_dir", type=str, default="output/test_mulitply_albedo/diffusionlight",
                        help="Directory to save the output images.")
    parser.add_argument("--scenes", default="14n_copyroom1,14n_copyroom6,14n_copyroom8,14n_copyroom10,14n_office12",  help="List of scene names to process.")
    parser.add_argument("--albedo_mode", type=str, choices=["white", "gray", "albedo"], default="albedo",help="Mode for albedo processing")
    args = parser.parse_args()
    args.scenes = args.scenes.split(",")
    return args

def to_linear_rgb(image, gamma = 2.4):
    """Convert an sRGB image to linear RGB."""
    linear_img = np.power(image, gamma)
    return linear_img

def render_with_albedo(shading, albedo):
    """Multiply shading with albedo and return the rendered image."""
    rendered = (albedo / np.pi) * shading 
    return rendered

def get_albedo_image(scene, albedo_dir, mode="albedo"):
    """Load the albedo image for a given scene."""
    albedo_path = os.path.join(albedo_dir, scene + ".png")
    albedo = skimage.io.imread(albedo_path)
    albedo = skimage.img_as_float(albedo)  # Convert to float if needed
    if mode == "white":
        albedo = np.ones_like(albedo) 
    elif mode == "gray":
        albedo = np.ones_like(albedo) * 0.5 
    return albedo

def main():
    args = parse_args()
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    for scene in args.scenes:
        shading_dir = os.path.join(args.shading_dir, scene)
        files = sorted(os.listdir(shading_dir))
        output_path = os.path.join(args.output_dir, scene)
        os.makedirs(output_path, exist_ok=True)

        albedo = get_albedo_image(scene, args.albedo_dir, mode=args.albedo_mode)
        albedo = to_linear_rgb(albedo, gamma=2.4)  # Convert to linear RGB

        for file in tqdm(files):
            shading = ezexr.imread(os.path.join(shading_dir, file))
            rendered = render_with_albedo(shading, albedo)
            rendered = tonemap(rendered, clip=True)[0]
            rendered = skimage.img_as_ubyte(rendered)
            output_file = os.path.join(args.output_dir, scene, file.replace('.exr', '.png'))
            skimage.io.imsave(output_file, rendered)
        

if __name__ == "__main__":
    main()