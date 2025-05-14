# from given spherical harmonic, we create 60 rotated speherical harmonics on given target scnee

import os 
import shutil
from tqdm.auto import tqdm

SOURCE_PATH = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/test/images"
scenes = sorted(os.listdir(SOURCE_PATH))


for scene in tqdm(scenes):
    # copy image 
    shutil.copytree(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/test/{scene}/shading_exr_perspective_v3_order6_marigold_v2",f"/pure/t1/datasets/laion-shading/v4/test/shadings_marigold/{scene}", dirs_exist_ok=True)
