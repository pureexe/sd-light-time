import os
import ezexr
import skimage
from tqdm.auto import tqdm
import numpy as np
from multiprocessing import Pool

SPLIT = "test"
PREDICT_DIR = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250217_albedo_optimization_v4/output/compute_albedo/{SPLIT}"
SOURCE_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT}/images"
NUM_THREADS = 20

def process_scene(scene):
    scene_root = os.path.join(PREDICT_DIR, scene, 'lightning_logs')
    versions = sorted(os.listdir(scene_root))
    out_dir = os.path.join(scene_root, versions[-1], 'shading_divide')  # from neural grapffer
    os.makedirs(out_dir, exist_ok=True)
    
    # Read albedo
    albedo_files = sorted(os.listdir(os.path.join(scene_root, versions[-1], 'albedo')))
    albedo_path = os.path.join(scene_root, versions[-1], 'albedo', albedo_files[-1])
    albedo = skimage.io.imread(albedo_path)
    albedo = skimage.img_as_float(albedo)
    
    max_values = []
    scene_image_dir = os.path.join(SOURCE_DIR, scene)
    image_files = sorted(os.listdir(scene_image_dir))
    
    for image_file in image_files:
        image = skimage.io.imread(os.path.join(SOURCE_DIR, scene, image_file))
        image = skimage.img_as_float(image)  # range [0,1]
        #shading = image / (albedo + 1e-6)
        shading = np.where(albedo < 1 / 255.0, 0, image / (albedo + 1e-6))

        max_values.append(shading.max())
        ezexr.imwrite(os.path.join(out_dir, image_file.replace('.jpg', '.exr')), shading)
    
    return max_values

def main():
    scenes = sorted(os.listdir(PREDICT_DIR))
    max_values = []
    
    with Pool(NUM_THREADS) as pool:
        for result in tqdm(pool.imap(process_scene, scenes), total=len(scenes)):
            max_values.extend(result)
    
    max_values = np.array(max_values)
    print("max values: ", max_values.max())
    print("max values mean: ", max_values.mean())

if __name__ == "__main__":
    main()
