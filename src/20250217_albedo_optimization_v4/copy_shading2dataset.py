import shutil 
from tqdm.auto import tqdm
import os 
from multiprocessing import pool
OUT_NAME = "control_shading_from_fitting_v3"
SOURCE_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/train"
TARGET_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"


def get_version_dir(scene_dir):
    all_versions = sorted(
        os.listdir(os.path.join(scene_dir, 'lightning_logs'))
    )
    lastest_version = all_versions[-1]
    return os.path.join(scene_dir, 'lightning_logs', lastest_version)


def main():
    scenes = sorted(os.listdir(SOURCE_DIR))
    in_dir = SOURCE_DIR
    for scene in tqdm(scenes):
        dataset_dir = get_version_dir(os.path.join(in_dir,scene))
        out_dir = os.path.join(TARGET_DIR, OUT_NAME, scene)
        shutil.copytree(
            os.path.join(dataset_dir, "shadings"),
            out_dir
        )
    

if __name__ == "__main__":
    main()