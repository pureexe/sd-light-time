# control_albedo_from_fitting_v2

import shutil 
from tqdm.auto import tqdm
import os 
from multiprocessing import pool
OUT_NAME = "control_albedo_from_fitting_v2"
SOURCE_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/test"
TARGET_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test"


def get_version_dir(scene_dir):
    all_versions = sorted(
        os.listdir(os.path.join(scene_dir, 'lightning_logs'))
    )
    lastest_version = all_versions[-1]
    return os.path.join(scene_dir, 'lightning_logs', lastest_version)

def get_lastest_albedo_path(dataset_dir):
    albedo_name = sorted(os.listdir(os.path.join(dataset_dir, 'albedo')))[-1]
    return os.path.join(os.path.join(dataset_dir, 'albedo', albedo_name))


def main():
    scenes = sorted(os.listdir(SOURCE_DIR))
    in_dir = SOURCE_DIR
    out_dir = os.path.join(TARGET_DIR, OUT_NAME)
    os.makedirs(out_dir,exist_ok=True)
    for scene in tqdm(scenes):
        dataset_dir = get_version_dir(os.path.join(in_dir,scene))
        albedo_path = get_lastest_albedo_path(dataset_dir)
        shutil.copy2(
            albedo_path,
            os.path.join(out_dir, scene+'.png')
        )
    

if __name__ == "__main__":
    main()