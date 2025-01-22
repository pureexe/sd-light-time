import os 
import torch
import numpy as np
import skimage
from sh_utils import get_shcoeff, cartesian_to_spherical, compute_background, get_ideal_normal_ball_z_up, get_uniform_rays, get_rotation_matrix_from_vectors_single
import ezexr
import time
from tonemapper import TonemapHDR


NUM_RAY = 100
OUTPUT_DIR = "output/order2"

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
    rays_rotation_matrix = np.zeros((H, W, 3, 3))
    for i in range(H):
        for j in range(W):
            rays_rotation_matrix[i,j] = get_rotation_matrix_from_vectors_single(np.array([0,0,1]), normal_map[i,j])

    print("Rotating rays")
    # rotate the rays to the normal map direction
    for i in range(H):
        for j in range(W):
            for k in range(num_ray):
                rays[i,j,k] = np.dot(rays_rotation_matrix[i,j], rays[i,j,k]) # shape [H,W, num_ray, 3]

    # compute n_dot_l
    print("Computing n_dot_l")
    n_dot_l = np.zeros((H,W,num_ray))
    for i in range(H):
        for j in range(W):
            for k in range(num_ray):
                n_dot_l[i,j,k] = np.dot(normal_map[i,j], rays[i,j,k])



    # look up the environment map to create an incoming light
    # 
    theta, phi = cartesian_to_spherical(rays)  # shape [H,W, num_ray]
    ## convert to [0,1] range
    theta = (theta + np.pi/2) / np.pi # now value is turn into [1,0]
    tehta = 1 - theta   # now value is turn into [0,1]
    phi = phi / (2*np.pi) # value is turn into [0,1]
    ### grid_sample use top-left as (-1,-1) and bottom-right as (1,1)
    theta = 2 * theta - 1
    phi = 2 * phi - 1
    theta_phi = np.stack([theta, phi], axis=-1) # shape [H,W, num_ray, 2]
    theta_phi = torch.tensor(theta_phi, dtype=torch.float32)
    # flatten H,W to the batch dimension
    theta_phi = theta_phi.reshape(H*W, num_ray, 2)[None] # shape [1, H*W, num_ray, 2]

    # convert environment map to tensor shape [1,3,H,W]
    envmap = torch.tensor(envmap, dtype=torch.float32)
    envmap = envmap.permute(2,0,1)
    envmap = envmap[None, ...] # shape [1,3,H,W]

    # look up the environment map using rays
    incoming = torch.nn.functional.grid_sample(envmap, theta_phi, align_corners=True) # shape [1,3,H*W,num_ray]
    # unflatten H,W
    incoming = incoming[0].permute(1,2,0).reshape(H,W,num_ray,3).numpy() # shape [H,W,num_ray,3]
 
    albedo = np.ones((H,W,3)) # currently use all bright albedo 
    reflectance = albedo / np.pi # lambertian reflectance is albedo/pi
    reflectance = reflectance[:,:,None] # shape [H,W,1,3]

    # compute shading 
    shading = reflectance * n_dot_l[...,None] * incoming
    shading = np.sum(shading, axis=-2) # shape [H,W,3]

    return shading

@torch.inference_mode()
def main():
    start_time = time.time()
    if False:
        image = get_envmap_from_file()
        shcoeff = get_shcoeff(image, Lmax=2)
        assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2 and shcoeff.shape[2] == 3 and shcoeff.shape[3] == 3 # make sure shape [3,2,3,3]

        # create envmap from spherical harmonic and save the result
        envmap = get_envmap_from_sh(shcoeff, lmax=2, image_width=512)
    else:
        envmap = get_envmap_from_file()
    if False:
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "envmap.png"), skimage.img_as_ubyte(envmap))

    # create normal ball and save the ball 
    normal_map, mask = get_ideal_normal_ball_z_up(256)
    if False:
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