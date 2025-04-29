import json
import os 

IN_PATH = '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/prompts_before.json'
OUT_PATH = '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/prompts.json'

with open(IN_PATH, 'r') as f:
    ori_prompt = json.load(f)

scenes = sorted(os.listdir('/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/test/images'))
new_prompts = {}
for scene in scenes:
    source = ori_prompt[f'{scene}/dir_0_mip2']
    for i in range(60):
        new_prompts[f'{scene}/dir_{i}_mip2'] = source 

with open(OUT_PATH, 'w') as f:
    json.dump(new_prompts, f, indent=4)