import ezexr
import os
from tqdm import tqdm
import skimage
import numpy as np
from multiprocessing import Pool, cpu_count

INPUT_DIR = "/pure/f1/datasets/multi_illumination/real_image_gt_shading/v0/train"
SHADING_DIR = "shadings"
ALBEDO_DIR = "albedos"
IMAGE_DIR = "images"
NUM_PROCESSES = 16  # Set to min(NUM_PROCESSES, cpu_count()) if you want it adaptive

def gamma_correct(image, gamma):
    """Apply gamma correction to an image."""
    return image ** gamma

def process_image(args):
    scene, image_name = args
    shading_path = os.path.join(INPUT_DIR, SHADING_DIR, scene, f"{image_name}.exr")
    albedo_path = os.path.join(INPUT_DIR, ALBEDO_DIR, scene, f"{image_name}.exr")
    image_path = os.path.join(INPUT_DIR, IMAGE_DIR, scene, f"{image_name}.jpg")

    if not os.path.exists(shading_path):
        print(f"Shading file does not exist: {shading_path}")
        return

    try:
        shading = ezexr.imread(shading_path)
        shading = np.clip(shading, 1e-8, np.inf)
        image = skimage.io.imread(image_path)
        image = skimage.img_as_float(image)
        linear_image = gamma_correct(image, 2.4)
        albedo = linear_image * np.pi / shading

        os.makedirs(os.path.dirname(albedo_path), exist_ok=True)
        ezexr.imwrite(albedo_path, albedo)

        try:
            os.chmod(albedo_path, 0o777)
            os.chmod(os.path.dirname(albedo_path), 0o777)
        except:
            pass
    except Exception as e:
        print(f"Error processing {scene}/{image_name}: {e}")

def main():
    image_dir = os.path.join(INPUT_DIR, IMAGE_DIR)
    queues = []
    scenes = sorted(os.listdir(image_dir))
    print("BUILDING INDEX")
    for scene in tqdm(scenes):
        scene_dir = os.path.join(image_dir, scene)
        image_names = sorted(name.replace('.jpg', '') for name in os.listdir(scene_dir) if name.endswith(".jpg"))
        queues.extend((scene, image_name) for image_name in image_names)

    print("PROCESSING IN PARALLEL")
    with Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(process_image, queues), total=len(queues)))

if __name__ == "__main__":
    main()
