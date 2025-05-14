# from given spherical harmonic, we create 60 rotated speherical harmonics on given target scnee

import subprocess

SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]

for scene in SCENES:
    subprocess.run([f'python create_rotate_shcoeff.py --source_shcoeff /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/shcoeff_perspective_v3_order100_main/dir_0_mip2.npy  --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}/shcoeff_perspective_v3_order6'], cwd='/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/create_rotate_dataset', shell=True)
