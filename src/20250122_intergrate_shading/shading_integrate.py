import os 
import torch
import numpy as np
import skimage
from sh_utils import get_shcoeff, cartesian_to_spherical, compute_background, get_ideal_normal_ball_z_up, get_uniform_rays, get_rotation_matrix_from_vectors_single
import ezexr
import time
from tonemapper import TonemapHDR


NUM_RAY = 10000
OUTPUT_DIR = "output/order2_newsample_10k"

def get_envmap_from_file():
    """
    Get environemnt map image
    Returns: 
    - np.ndarray: image in HWC format in range [0,1]
    """
    image = skimage.io.imread("data/coordinates_z_up.png")[...,:3]
    image = skimage.img_as_float(image)
    return image

def get_envmap_from_sh(shcoeff, lmax=2, image_width=512):
    """
    Get environemnt map image
    Returns: 
    - np.ndarray: image in HWC format in range [0,1]
    """
    image =  compute_background(shcoeff, lmax=lmax, image_width=image_width)
    image = np.clip(image, 0, 1)
    return image

# TODO: need unit test for this function
def rotate_rays(rays_rotation_matrix, rays):
    """
    Rotate rays
    Parameters:
    - rays_rotation_matrix (np.ndarray): rotation matrix in [H,W,3,3]
    - rays (np.ndarray): rays in [H,W,num_ray,3]
    Returns:
    - np.ndarray: rotated rays in [H,W,num_ray,3]
    """
    H, W, num_ray = rays.shape[:3]
    for i in range(H):
        for j in range(W):
            for k in range(num_ray):
                rays[i,j,k] = np.dot(rays_rotation_matrix[i,j], rays[i,j,k])
    return rays

def get_rotation_matrix_from_normal(normal_map):
    """
    Get rotation matrix from ray
    UNIT TESTED
    Parameters:
    - normal_map (np.ndarray): normal_map in [H,W,3]
    Returns:
    - np.ndarray: rotation matrix in [H,W,3,3]
    """
    H, W = normal_map.shape[:2]
    rotation_matrix = np.zeros((H, W, 3, 3))
    for i in range(H):
        for j in range(W):
            rotation_matrix[i,j] = get_rotation_matrix_from_vectors_single(np.array([0,0,1]), normal_map[i,j])
    return rotation_matrix

def get_ndotl(normal_map, rays):
    """
    Compute n dot l
    UNIT TESTED
    Parameters:
    - normal_map (np.ndarray): normal map in [H,W,3]
    - rays (np.ndarray): rays in [H,W,num_ray,3]
    Returns:
    - np.ndarray: n dot l in [H,W,num_ray]
    """
    H, W = normal_map.shape[:2]
    num_ray = rays.shape[2]
    n_dot_l = np.zeros((H,W,num_ray))
    for i in range(H):
        for j in range(W):
            for k in range(num_ray):
                n_dot_l[i,j,k] = np.dot(normal_map[i,j], rays[i,j,k])

    n_dot_l = np.maximum(0, n_dot_l) # make sure n_dot_l is positive
    return n_dot_l

def get_incoming_coords(rays):
    """
    Get incoming coordinates
    UNIT TESTED (By get_incoming_light)
    Parameters:
    - rays (np.ndarray): rays in [H,W,num_ray,3]
    Returns:
    - np.ndarray: incoming coordinates in [H,W,num_ray,2]
    """
    theta, phi = cartesian_to_spherical(rays)  # shape [H,W, num_ray]
    ## convert to [0,1] range
    theta = (theta + np.pi/2) / np.pi # now value is turn into [1,0]
    theta = 1 - theta   # now value is turn into [0,1]
    phi = phi / (2*np.pi) # value is turn into [0,1]
    ### grid_sample use top-left as (-1,-1) and bottom-right as (1,1)
    theta = 2 * theta - 1
    phi = 2 * phi - 1
    phi_theta = np.stack([phi, theta], axis=-1) # shape [H,W, num_ray, 2]
    return phi_theta
        
def get_incoming_light(envmap, rays):
    """
    get incoming light direction from given ray direction
    UNIT TESTED
    Parameters:
    - envmap (np.ndarray): environment map in [H,W,3]
    - rays (np.ndarray): rays in [H,W,num_ray,3]
    Returns:
    - np.ndarray: incoming light in [H,W,num_ray,3]
    """
    H, W, num_ray = rays.shape[:3]

    phi_theta =  get_incoming_coords(rays) # shape [H,W, num_ray, 2]
    phi_theta = torch.tensor(phi_theta, dtype=torch.float32)
    # flatten H,W to the batch dimension
    phi_theta = phi_theta.reshape(H*W, num_ray, 2)[None] # shape [1, H*W, num_ray, 2]


    # convert environment map to tensor shape [1,3,H,W]
    envmap = torch.tensor(envmap, dtype=torch.float32)
    envmap = envmap.permute(2,0,1)
    envmap = envmap[None, ...] # shape [1,3,H,W]

    # look up the environment map using rays
    incoming = torch.nn.functional.grid_sample(envmap, phi_theta, align_corners=True) # shape [1,3,H*W,num_ray]
    # unflatten H,W
    incoming = incoming[0].permute(1,2,0).reshape(H,W,num_ray,3).numpy() # shape [H,W,num_ray,3]
 
    return incoming

# TODO: need a unit test for this function
def get_albedo(H,W):
    """
    Get albedo map
    Parameters:
    - H (int): height
    - W (int): width
    Returns:
    - np.ndarray: albedo map in [H,W,3]
    """
    albedo = np.ones((H,W,3)) # currently use all bright albedo 
    return albedo

# TODO: need a unit test for this function
def compute_diffuse_brdf(albedo, n_dot_l, incoming):
    """
    Compute diffuse BRDF
    Parameters:
    - albedo (np.ndarray): albedo in [H,W,3]
    - n_dot_l (np.ndarray): n dot l in [H,W,num_ray]
    - incoming (np.ndarray): incoming light in [H,W,num_ray,3]
    Returns:
    - np.ndarray: diffuse BRDF in [H,W,num_ray,3]
    """
    reflectance = albedo / np.pi # lambertian reflectance is albedo/pi
    reflectance = reflectance[:,:,None] # shape [H,W,1,3]
    diffuse_brdf = reflectance * n_dot_l[...,None] * incoming
    shading = np.sum(diffuse_brdf, axis=-2) # shape [H,W,3]
    return shading


@torch.inference_mode()
def render_shading_from_normal(normal_map, envmap, num_ray = NUM_RAY):
    """
    render shading from normal map 
    Parameters:
    - normal_map (np.ndarray): normal map in [H,W,3]
    - envmap (np.ndarray): environment map in [H,W,3]
    - num_ray (int): number of ray to sample
    Returns:
    - np.ndarray: shading image in [H,W,3] (this is HDR image, might go over 1)
    """
    H, W = normal_map.shape[:2]
    print("Getting rays")
    # sample rays on the hemisphere. the middle ray is currently pointing to [0,0,1]
    rays = get_uniform_rays(normal_map.shape[0], normal_map.shape[1], num_ray) 

    # we have rotate the rays to the normal map direction
    print("Computing rotation matrix")
    rays_rotation_matrix = get_rotation_matrix_from_normal(normal_map)

    # rotate the rays to the normal map direction
    print("Rotating rays")
    rays = rotate_rays(rays_rotation_matrix, rays)  
    
    # compute n_dot_l
    print("Computing n_dot_l")
    n_dot_l = get_ndotl(normal_map, rays)

    # get incoming light
    print("Getting incoming light")
    incoming = get_incoming_light(envmap, rays)

    # get albedo
    print("Getting albedo")
    albedo = get_albedo(H,W) # currently use all bright albedo 
    
    # compute shading
    print("Computing shading")
    shading = compute_diffuse_brdf(albedo, n_dot_l, incoming)

    return shading

@torch.inference_mode()
def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if True:
        image = get_envmap_from_file()
        shcoeff = get_shcoeff(image, Lmax=2)
        assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2 and shcoeff.shape[2] == 3 and shcoeff.shape[3] == 3 # make sure shape [3,2,3,3]

        # create envmap from spherical harmonic and save the result
        envmap = get_envmap_from_sh(shcoeff, lmax=2, image_width=512)
    else:
        envmap = get_envmap_from_file()

    if True:
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "envmap.png"), skimage.img_as_ubyte(envmap))

    # create normal ball and save the ball 
    normal_map, mask = get_ideal_normal_ball_z_up(256)
    if True:
        normal_ball_image = (normal_map + 1 / 2.0)
        normal_ball_image = skimage.img_as_ubyte(np.clip(normal_ball_image, 0, 1))
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "normal_ball.png"), normal_ball_image)

    # render shading from normal map
    shading = render_shading_from_normal(normal_map, envmap)
    ezexr.imwrite(os.path.join(OUTPUT_DIR, "rendered_shading.exr"), shading)
    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    tonemapped_shading, _, _ = tonemap(shading)
    skimage.io.imsave(os.path.join(OUTPUT_DIR, "tonemapped_shading.png"), skimage.img_as_ubyte(tonemapped_shading))
    print(f"Time elapsed: {time.time() - start_time}")



if __name__ == "__main__":
    main()