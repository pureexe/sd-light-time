import shutil
from tqdm.auto import tqdm
import os
from multiprocessing import Pool, cpu_count

OUT_NAME = "control_shading_from_fitting_v3_exr"
SOURCE_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/train"
TARGET_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
NUM_WORKERS = 20  # Number of parallel processes

def get_version_dir(scene_dir):
    all_versions = sorted(os.listdir(os.path.join(scene_dir, 'lightning_logs')))
    latest_version = all_versions[-1]
    return os.path.join(scene_dir, 'lightning_logs', latest_version)

def copy_scene(scene):
    try:
        dataset_dir = get_version_dir(os.path.join(SOURCE_DIR, scene))
        out_dir = os.path.join(TARGET_DIR, OUT_NAME, scene)
        shutil.copytree(os.path.join(dataset_dir, "shading_exr"), out_dir)
        return True
    except Exception as e:
        return f"Error copying {scene}: {e}"

def main():
    scenes = sorted(os.listdir(SOURCE_DIR))
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(copy_scene, scenes), total=len(scenes)))
        
        # Check for any errors
        errors = [res for res in results if isinstance(res, str)]
        if errors:
            print("Errors occurred:")
            for error in errors:
                print(error)

if __name__ == "__main__":
    main()
