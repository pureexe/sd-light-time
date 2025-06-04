import os 
import ezexr
import skimage
import numpy as np 
from tqdm.auto import tqdm
from tonemapper import TonemapHDR

ALBEDO_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_albedo_from_fitting_v2" #contain png file
SHADING_DIR = "/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train/shadings/"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250527_inspect_shading_value/multiply_gray/leastsquare"
SCENES = [
    "14n_copyroom1",
    "14n_copyroom6",
    "14n_copyroom8",
    "14n_copyroom10",
    "14n_office12"
]
ALBEDO_MODE = "GRAY"

def to_linear_rgb(image, gamma = 2.4):
    """Convert an sRGB image to linear RGB."""
    linear_img = np.power(image, gamma)
    return linear_img

def main():
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    for scene in SCENES:
        shading_path = os.path.join(SHADING_DIR, scene)
        files = sorted(os.listdir(shading_path))
        output_path = os.path.join(OUTPUT_DIR, scene)
        os.makedirs(output_path, exist_ok=True)
        albedo_path = os.path.join(ALBEDO_DIR, scene+ ".png")
        albedo = skimage.io.imread(albedo_path)
        albedo = skimage.img_as_float(albedo)  # Convert to float if needed
        #albedo = np.ones_like(albedo) * 0.5 / np.pi #gray color
        if ALBEDO_MODE == "WHITE":
            albedo = np.ones_like(albedo) 
        if ALBEDO_MODE == "GRAY":
            albedo = np.ones_like(albedo) * 0.5 


        for file in tqdm(files):
            shading_file = ezexr.imread(os.path.join(shading_path, file))
            rendered = albedo * shading_file 
            rendered = np.clip(rendered, 0, 1)
            rendered = skimage.img_as_ubyte(rendered)
            output_file = os.path.join(OUTPUT_DIR, scene, file.replace('.exr', '.png'))
            skimage.io.imsave(output_file, rendered)
        

if __name__ == "__main__":
    main()