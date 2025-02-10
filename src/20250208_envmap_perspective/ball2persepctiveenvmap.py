# convert the ball to environment map, lat, long format 

import numpy as np
from PIL import Image
import skimage
import time
import torch
import argparse 
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import os
import shutil

try:
    import ezexr
except:
    pass

# python ball2persepctiveenvmap.py --ball_dir /ist/ist-share/vision/relight/datasets/multi_illumination_train_mip2_exr --envmap_dir /ist/ist-share/vision/relight/datasets/multi_illumination/unused/exr_envmap_train_mip2_exr_v3 --fov_dir /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_focallength --threads 20
# python ball2persepctiveenvmap.py --ball_dir /ist/ist-share/vision/relight/datasets/multi_illumination_train_mip2_jpg --envmap_dir /ist/ist-share/vision/relight/datasets/multi_illumination/unused/ldr_envmap_train_mip2_jpg_v3 --fov_dir /ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/metric_focallength


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, required=True ,help='directory that contain the image') 
    parser.add_argument("--envmap_dir", type=str, required=True ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--fov_dir", type=str, required=True ,help='field of view directory. contain match npy file ') #dataset name or directory 
    parser.add_argument("--envmap_height", type=int, default=256, help="size of the environment map height in pixel (height)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--threads", type=int, default=8, help="num thread for pararell processing")
    return parser

def copy_fov_from_neightbour(args, filename):
    fov_path = os.path.join(args.fov_dir, filename)
    current_dir = os.path.dirname(fov_path)
    # find other file
    other_fov = [f for f in sorted(os.listdir(current_dir)) if f.endswith('.npy')]
    shutil.copy2(os.path.join(current_dir, other_fov[0]),fov_path)

def get_fov(args, filename):
    IMAGE_WIDTH = 512
    fov_path = os.path.join(args.fov_dir, filename)
    fov_px = np.load(fov_path)
    fov_rad = 2 * np.arctan2(IMAGE_WIDTH, 2*fov_px)
    return fov_rad

def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION (x-forward, y-right, z-up)
    Create the grid of environment map that contain the position in sperical coordinate
    # Top left is (theta=-0.5,phi=-1) and bottom right is (theta=0.5, phi=1)
    Top left is (theta=-pi/2,phi=-pi) and bottom right is (theta=pi/2, phi=pi)
    """    
    
    # theta = torch.linspace(0.5, -0.5, size)
    # phi = torch.linspace(-1, 1, size * 2)

    theta = torch.linspace(np.pi / 2, -np.pi / 2, size)
    phi = torch.linspace(-np.pi, np.pi, size * 2)


    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='ij')     
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

def convert_theta_from_ball_surface_to_inside_envmap(theta):
    return (-theta + 3 * np.pi) % (2 * np.pi)

def apply_black_outside_ball(image: np.ndarray) -> np.ndarray:
    """
    Masks out the image outside a central ball (circle) by setting it to black.
    
    Parameters:
        image (np.ndarray): Input image of shape (H, W, 3)
    
    Returns:
        np.ndarray: Masked image where outside the circle is black.
    """
    H, W, _ = image.shape
    
    # Compute the center and radius of the circle
    center_x, center_y = W // 2, H // 2
    radius = min(center_x, center_y)
    
    # Create a coordinate grid
    Y, X = np.ogrid[:H, :W]
    
    # Create mask for the circle
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    
    # Apply mask to the image
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    return masked_image

# blender format is wrap from outside to inside    
def process_image(args: argparse.Namespace, file_name: str):

    if isinstance(file_name, list):
        npy_name = file_name[1]
        file_name = file_name[0]
    else:
        npy_name = file_name

    I = np.array([1,0, 0])
 
    # check if exist, skip!
    envmap_output_path = os.path.join(args.envmap_dir, file_name)
    if os.path.exists(envmap_output_path):
        return None
    
    # read ball image 
    ball_path = os.path.join(args.ball_dir, file_name)
    if file_name.endswith(".exr"):
        try:
            ball_image = ezexr.imread(ball_path)
        except:
            print(ball_path)
            print("FAILED TO READ EXR")
            return None
    else:
        try:
            ball_image = skimage.io.imread(ball_path)
            ball_image = skimage.img_as_float(ball_image)
        except:
            return None

    # apply black mark on region outside ball
    #ball_image = apply_black_outside_ball(ball_image)

    # get field of view
    try:
        fov = get_fov(args, npy_name)
    except:
        print("FOV FAILED")
        copy_fov_from_neightbour(args, npy_name)
        return None
    nFOV = np.pi - fov

    # compute  normal map that create from reflect vector
    env_grid = create_envmap_grid(args.envmap_height * args.scale)  # [phi [-pi,pi], theta [pi/2, -pi/2]]
    H,W = env_grid.shape[:2]

    theta, phi = env_grid[...,0], env_grid[...,1]
    reflect_vec = get_cartesian_from_spherical(theta, phi) # (x-forward, y-right, z-up) range [-1,1]

    normal = get_normal_vector(I[None,None], reflect_vec) # (x-forward, y-right, z-up) range [-1,1]

    
    # We ignore X axis because it represent forward. (y-right, z-up) range [-1,1]
    #y,z = reflect_vec[...,1], reflect_vec[...,2]
    y,z = normal[...,1], normal[...,2]
    
    # Normalize FOV (radius) that the limit of FOV will be on the border
    r = (nFOV) / (np.pi)  #
    #u,v is the position on the ball 
    u = y / r 
    v = z / r

    mask = (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)


    pos = np.concatenate([u[...,None],v[...,None]], axis=-1)

    # since Z-UP (top 1, bottom -1) but torch grid sample is top-1 bottom 1 (is z-down) we flip only z axis 
    BLENDER_CONVENTION = True
    if BLENDER_CONVENTION:
        pos  = -pos
    else:
        pos[...,1] = -pos[...,1]

    
    env_map = None
    
    # using pytorch method for bilinear interpolation
    with torch.no_grad():
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos)[None].float()

        # convert ball to support pytorch
        ball_image = torch.from_numpy(ball_image[None]).float()
        ball_image = ball_image.permute(0,3,1,2) # [1,3,H,W]
        
        env_map = torch.nn.functional.grid_sample(ball_image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        env_map = env_map[0].permute(1,2,0).numpy()

    env_map = env_map * mask[...,None]
                
    env_map_default = skimage.transform.resize(env_map, (args.envmap_height, args.envmap_height*2), anti_aliasing=True)
    if file_name.endswith(".exr"):
        ezexr.imwrite(envmap_output_path, env_map_default.astype(np.float32))
    else:
        env_map_default = skimage.img_as_ubyte(env_map_default)        
        skimage.io.imsave(envmap_output_path, env_map_default)
    return None


def main():
    
    # running time measuring
    start_time = time.time()        

    # load arguments
    args = create_argparser().parse_args()
    
    # make output directory if not exist
    os.makedirs(args.envmap_dir, exist_ok=True)
    
    # get all file in the directory
    scenes = sorted(os.listdir(args.ball_dir))

    files = []

    for scene in scenes:
        os.makedirs(os.path.join(args.envmap_dir, scene,'probes'), exist_ok=True)
        for light_id in range(25):
            files.append(
                [
                    os.path.join(scene,'probes',f'dir_{light_id}_chrome256.exr'),
                    os.path.join(scene, f'dir_{light_id}_mip2.npy'),
                ]
            ) 

    # create partial function for pararell processing
    process_func = partial(process_image, args)

    # process_func(files[0])
    # exit()

    
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)        

    
   
if __name__ == "__main__":
    main()    
    