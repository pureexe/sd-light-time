import numpy as np
import json

#SHOULD ABOUT THE FOCAL LENGTH

def depth_to_obj(depth, focal, obj_path, json_path):
    """
    Converts a depth map to a 3D mesh, saves it as an OBJ file, and saves additional metadata as a JSON file.

    Parameters:
        depth (np.ndarray): A 2D array of metric depth values with shape [H, W].
        focal (float): Focal length of the camera.
        obj_path (str): File path to save the OBJ file.
        json_path (str): File path to save the JSON metadata file.
    """
    # Get height (H) and width (W) of the depth map
    H, W = depth.shape

    # Create a grid of pixel coordinates
    x = np.arange(W)
    y = np.arange(H)
    xv, yv = np.meshgrid(x, y)

    # Convert pixel coordinates to normalized image coordinates
    x_norm = (xv - W / 2) / focal
    y_norm = (H / 2 - yv) / focal  # Flip Y-axis

    # Calculate 3D coordinates
    Z = depth
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

    # Adjust Z values
    Z = Z - z_offset

    # Write to OBJ file
    with open(obj_path, 'w') as f:
        # Write vertices
        for i in range(len(X)):
            f.write(f"v {X[i]} {Y[i]} {-Z[i]}\n")

        # Write faces
        for y in range(H - 1):
            for x in range(W - 1):
                idx0 = y * W + x + 1
                idx1 = idx0 + 1
                idx2 = idx0 + W
                idx3 = idx2 + 1
                f.write(f"f {idx0} {idx2} {idx1}\n")
                f.write(f"f {idx1} {idx2} {idx3}\n")

    # Save metadata to JSON
    metadata = {
        "focal": float(focal),
        "z_offset": float(z_offset)
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# Example usage
if __name__ == "__main__":
    # Example depth map
    #depth = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    #focal = 1.0
    depth = np.load("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_depth/14n_copyroom1/dir_0_mip2.npy")
    focal = np.load("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_focallength/14n_copyroom1/dir_0_mip2.npy")
    obj_path = "14n_copyroom1_dir_0.obj"
    json_path = "14n_copyroom1_dir_0.json"
    depth_to_obj(depth, focal, obj_path, json_path)
