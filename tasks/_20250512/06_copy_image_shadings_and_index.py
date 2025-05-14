# from given spherical harmonic, we create 60 rotated speherical harmonics on given target scnee

import shutil
from tqdm.auto import tqdm
SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]

for scene in tqdm(SCENES):
    # copy image 
    #shutil.copytree(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/images",f"/pure/t1/datasets/laion-shading/v4/rotate/images/{scene}", dirs_exist_ok=True)
    # copy shading
    shutil.copytree(f"/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/shading_exr_perspective_v3_order6",f"/pure/t1/datasets/laion-shading/v4/rotate/shadings_marigold/{scene}", dirs_exist_ok=True)
    # copy index 
    # shutil.copy2(f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_test_scenes/{scene}_rotate.json","")