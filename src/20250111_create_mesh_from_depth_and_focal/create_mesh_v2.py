import numpy as np
import json
import os 
from tqdm.auto import tqdm
from multiprocessing import Pool

def depth_to_obj(depth, focal, obj_path, json_path):
    """
    Converts a depth map to a 3D mesh, saves it as an OBJ file, and saves additional metadata as a JSON file.

    Parameters:
        depth (np.ndarray): A 2D array of metric depth values with shape [H, W]. (in unit of meters)
        focal (float): Focal length of the camera. (in unit of pixels)
        obj_path (str): File path to save the OBJ file.
        json_path (str): File path to save the JSON metadata file.
    """
    # Get height (H) and width (W) of the depth map
    H, W = depth.shape # H=512, W=512

    # Create a grid of pixel coordinates
    x = np.arange(W) + 0.5 # generate 0.5 to W-0.5 (511.5) to be center of pixel
    y = np.arange(H) + 0.5

    # we pad 0 and 512 pixel 
    x = np.concatenate(([0], x, [W]))
    y = np.concatenate(([0], y, [H]))

    xv, yv = np.meshgrid(x, y)
 
    # Convert pixel coordinates to normalized image coordinates
    x_norm = (xv - (W / 2)) / focal 
    y_norm = ((H / 2) - yv) / focal  # Flip Y-axis

    # Calculate 3D coordinates
    Z = np.pad(depth, pad_width=1, mode='edge') # pad 4 edge 
    X = x_norm * Z
    Y = y_norm * Z

    # Flatten arrays for OBJ format
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Calculate near and far planes and offset
    near_plane = np.percentile(Z, 1)
    far_plane = np.percentile(Z, 99)
    z_offset = float((far_plane + near_plane) / 2)

    # Adjust Z values, So model is centered at 0,0,0
    Z = Z - z_offset

    # Write to OBJ file
    with open(obj_path, 'w') as f:
        # Write vertices
        for i in range(len(X)):
            f.write(f"v {X[i]} {Y[i]} {-Z[i]}\n")

        # Write faces
        for y in range(H + 1):
            for x in range(W + 1):
                idx0 = y * (W+2) + x + 1 # top-left
                idx1 = idx0 + 1 # top-right
                idx2 = idx0 + (W+2) # bottom-left
                idx3 = idx2 + 1 # bottom-right
                f.write(f"f {idx0} {idx2} {idx1}\n") # top-left, bottom-left, top-right
                f.write(f"f {idx1} {idx2} {idx3}\n") # top-right, bottom-left, bottom-right

    # Save metadata to JSON
    metadata = {
        "focal": float(focal),
        "z_offset": float(z_offset)
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)


# Example usage
# if __name__ == "__main__":
#     depth = np.load("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_depth/14n_copyroom1/dir_0_mip2.npy")
#     focal = np.load("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_focallength/14n_copyroom1/dir_0_mip2.npy")
#     obj_path = "output/14n_copyroom1_dir_0.obj"
#     json_path = "output/14n_copyroom1_dir_0.json"
#     depth_to_obj(depth, focal, obj_path, json_path)



# if __name__ == "__main__":
#     DEPTH_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/metric_depth"
#     FOCAL_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/metric_focallength"
#     OUT_MESH_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/test/mesh"
#     OUT_JSON_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/test/focal_json"
#     os.makedirs(OUT_JSON_DIR, exist_ok=True)
#     os.makedirs(OUT_MESH_DIR, exist_ok=True)
#     scenes = sorted(os.listdir(DEPTH_DIR))
#     for scene in tqdm(scenes):
#         obj_path = f"{OUT_MESH_DIR}/{scene}.obj"
#         json_path = f"{OUT_JSON_DIR}/{scene}.json"
#         depth = np.load(f"{DEPTH_DIR}/{scene}/dir_0_mip2.npy")
#         focal = np.load(f"{FOCAL_DIR}/{scene}/dir_0_mip2.npy")
#         depth_to_obj(depth, focal, obj_path, json_path)


DEPTH_DIR = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/metric_depth"
FOCAL_DIR = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/metric_focallength"
OUT_MESH_DIR = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/mesh"
OUT_JSON_DIR = "/ist/ist-share/vision/relight/datasets/unsplash-lite/train/focal_json"
    

def process(scene):
    obj_path = f"{OUT_MESH_DIR}/{scene}.obj"
    json_path = f"{OUT_JSON_DIR}/{scene}.json"
    if os.path.exists(obj_path) and os.path.exists(json_path):
        return None
    depth = np.load(f"{DEPTH_DIR}/{scene}")
    focal = np.load(f"{FOCAL_DIR}/{scene}")
    depth_to_obj(depth, focal, obj_path, json_path)

if __name__ == "__main__":
    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUT_MESH_DIR, exist_ok=True)
    scenes = sorted(os.listdir(DEPTH_DIR))
    with Pool(40) as p:
        r = list(tqdm(p.imap(process, scenes), total=len(scenes)))
  

# if __name__ == "__main__":
#     os.makedirs(OUT_JSON_DIR, exist_ok=True)
#     os.makedirs(OUT_MESH_DIR, exist_ok=True)
#     scenes = sorted(os.listdir(DEPTH_DIR))
#     for scene in tqdm(scenes):
#         obj_path = f"{OUT_MESH_DIR}/{scene}.obj"
#         json_path = f"{OUT_JSON_DIR}/{scene}.json"
#         depth = np.load(f"{DEPTH_DIR}/{scene}")
#         focal = np.load(f"{FOCAL_DIR}/{scene}")
#         depth_to_obj(depth, focal, obj_path, json_path)