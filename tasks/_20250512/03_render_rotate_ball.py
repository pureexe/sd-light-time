# from given spherical harmonic, we create 60 rotated speherical harmonics on given target scnee

import subprocess

SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]

for scene in SCENES:
    subprocess.run([f'python efficient_sh_general_rotate.py --input_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate --output_dir /ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate --threads 32 --coeff_dir shcoeff_perspective_v3_order6 --use_ball 1 --shading_dir shading_exr_perspective_v3_order6_ball --vizmax_dir shading_exr_perspective_v3_order6_ball_viz_max --vizldr_dir shading_exr_perspective_v3_order6_ball_viz_ldr'], cwd='/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/scripts/efficient_rendering', shell=True)
