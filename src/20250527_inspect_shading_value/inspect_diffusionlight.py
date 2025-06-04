import os 
import ezexr

ALBEDO_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_albedo_from_fitting_v2" #contain png file
SHADING_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train//control_shading_from_fitting_v3_exr"
TARGET_SCENE = "14n_copyroom10"

def main():
    shading_path = os.path.join(SHADING_DIR, TARGET_SCENE)
    files = sorted(os.listdir(shading_path))
    for filename in files:
        if filename.endswith(".exr"):
            exr_image = ezexr.imread(os.path.join(DATASET_PATH, filename))
            print(filename)
            print(exr_image.min())
            print(exr_image.max())

if __name__ == "__main__":
    main()