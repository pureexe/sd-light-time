
import depth_pro
from PIL import Image
import depth_pro
from tqdm.auto import tqdm
import os

ROOT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test"

def convert_filename_to_npy(filename):
    return filename.replace(".jpg", ".npy").replace(".png", ".npy")

def main():

    # list all scenes 
    queues = []
    scenes = sorted(os.listdir(os.path.join(ROOT_DIR, "images")))
    for scene in scenes:
        files = sorted(os.listdir(os.path.join(ROOT_DIR, "images", scene)))
        for filename in files:
            queues.append(scene + "/" + filename)


    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    pbar = tqdm(queues)
    pbar.set_description(f"")

    for q in pbar:
        depth_dir = os.path.join(ROOT_DIR, "metric_depth", dir_name)
        depth_path = os.path.join(depth_dir, npy_name)
        if os.path.exists(depth_path):
            continue
    
        pbar.set_postfix(item=f"{q}")


        # Load and preprocess an image.
        image_path = os.path.join(ROOT_DIR, "images", q)
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels 

        dir_name = q.split("/")[0]
        npy_name = convert_filename_to_npy(q.split("/")[1])

        # Save the depth map.
        
        os.makedirs(depth_dir, exist_ok=True)
        os.chmod(depth_dir, 0o777)
        
        os.chmod(depth_path, 0o777)
        np.save(depth_path, depth)

        # Save the focal length.
        focal_dir = os.path.join(ROOT_DIR, "metric_focallength", dir_name)
        os.makedirs(focal_dir, exist_ok=True)
        os.chmod(focal_dir, 0o777)
        focal_path = os.path.join(focal_dir, npy_name)
        os.chmod(focal_path, 0o777)
        np.save(focal_path, focallength_px)


if __name__ == "__main__":
    main()