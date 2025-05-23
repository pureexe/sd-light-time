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

try:
    import ezexr
except:
    pass

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, required=True ,help='directory that contain the image') 
    parser.add_argument("--envmap_dir", type=str, required=True ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--envmap_height", type=int, default=256, help="size of the environment map height in pixel (height)")
    parser.add_argument("--scale", type=int, default=4, help="scale factor")
    parser.add_argument("--threads", type=int, default=8, help="num thread for pararell processing")
    return parser

def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION (x-forward, y-right, z-up)
    Create the grid of environment map that contain the position in sperical coordinate
    Top left is (theta=pi/2,phi=0) and bottom right is (theta=-pi/2, phi=2pi)
    """    
    
    theta = torch.linspace(np.pi / 2, -np.pi / 2, size)
    phi = torch.linspace(0, 2*np.pi, size * 2)
    
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

# blender format is wrap from outside to inside    
def process_image(args: argparse.Namespace, file_name: str):
    I = np.array([1,0, 0])
 
    # check if exist, skip!
    envmap_output_path = os.path.join(args.envmap_dir, file_name)
    if os.path.exists(envmap_output_path):
        return None
    
    # read ball image 
    ball_path = os.path.join(args.ball_dir, file_name)
    if file_name.endswith(".exr"):
        ball_image = ezexr.imread(ball_path)
    else:
        try:
            ball_image = skimage.io.imread(ball_path)
            ball_image = skimage.img_as_float(ball_image)
        except:
            return None

    # compute  normal map that create from reflect vector
    env_grid = create_envmap_grid(args.envmap_height * args.scale)  # [phi 0-2pi, theta pi/2-(-pi/2)]
    theta, phi = env_grid[...,0], env_grid[...,1]
    #theta = convert_theta_from_ball_surface_to_inside_envmap(theta)
    reflect_vec = get_cartesian_from_spherical(theta, phi) # (x-forward, y-right, z-up) range [-1,1]
    normal = get_normal_vector(I[None,None], reflect_vec) # (x-forward, y-right, z-up) range [-1,1]
    
    # turn from normal map to range [0,1] # (x-forward, y-right, z-up)
    pos = (normal + 1.0) / 2

    # We ignore X axis because it represent forward. (y-right, z-up) range [0,1]
    pos = pos[...,1:] 

    # since Z-UP (top 1, bottom -1) but torch grid sample is top-1 bottom 1 (is z-down) we flip only z axis
    # pos[...,1] = 1.0 - pos[...,1]
    pos  = 1.0 - pos

    # since pos grid_sample is using
    pos = pos * 2 - 1 # convert to range [-1,1]    
    
    env_map = None
    
    # using pytorch method for bilinear interpolation
    with torch.no_grad():
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos)[None].float()

        # convert ball to support pytorch
        ball_image = torch.from_numpy(ball_image[None]).float()
        ball_image = ball_image.permute(0,3,1,2) # [1,3,H,W]
        
        env_map = torch.nn.functional.grid_sample(ball_image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        env_map = env_map[0].permute(1,2,0).numpy()
                
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
    files = sorted(os.listdir(args.ball_dir))
    
    # create partial function for pararell processing
    process_func = partial(process_image, args)
    
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)        

    
   
if __name__ == "__main__":
    main()    
    