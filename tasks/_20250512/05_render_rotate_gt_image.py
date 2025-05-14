# from given spherical harmonic, we create 60 rotated speherical harmonics on given target scnee

import subprocess

SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]

for scene in SCENES:
    subprocess.run([f'python create_gt_image.py --source_image /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/rotate/images/{scene}/dir_0_mip2.png --shadings_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/shading_exr_perspective_v3_order6 --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/images'], cwd='/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/create_rotate_dataset', shell=True)
