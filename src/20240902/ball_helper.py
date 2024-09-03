import torch 
import numpy as np 
import skimage
import os


def create_circle_tensor(num_pixel, circle_size):
    """
    Create a PyTorch tensor with a circle in the middle.
    
    Args:
        num_pixel (int): The size of the tensor (num_pixel x num_pixel).
        circle_size (int): The diameter of the circle.
        
    Returns:
        torch.Tensor: A tensor with a circle in the middle (1.0 inside, 0.0 outside).
    """
    # Create a tensor of zeros
    tensor = torch.zeros((num_pixel, num_pixel), dtype=torch.float32)
    
    # Compute the center and radius of the circle
    center = num_pixel // 2
    radius = circle_size // 2
    
    # Define the grid
    y, x = torch.meshgrid(torch.arange(num_pixel), torch.arange(num_pixel))
    
    # Calculate the distance from the center
    distance_from_center = torch.sqrt((x - center)**2 + (y - center)**2)
    
    # Update tensor values based on the distance from the center
    tensor[distance_from_center <= radius] = 1.0
    
    return tensor

def get_reflection_vector_map(I: np.array, N: np.array):
    """
    UNIT-TESTED
    Args:
        I (np.array): Incoming light direction #[None,None,3]
        N (np.array): Normal map #[H,W,3]
    @return
        R (np.array): Reflection vector map #[H,W,3]
    """
    
    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = (I[...,None,:] @ N[...,None])[...,0]
    R = I - 2 * dot_product * N
    return R


def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)


def get_ideal_normal_ball(size):
    
    """
    UNIT-TESTED
    BLENDER CONVENTION
    X: forward
    Y: right
    Z: up
    
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(1, -1, size)
    
    #use indexing 'xy' torch match vision's homework 3
    y,z = torch.meshgrid(y, z ,indexing='xy') 
    
    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    x = torch.sqrt(x)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask


def envmap2chromeball(env_map):
    """ convert envmap to image with ball in the middle

    """

    assert env_map.shape[1] == 3

    normal_ball, _ = get_ideal_normal_ball(512)

    # verify that x of normal is in range [0,1]
    assert normal_ball[:,:,0].min() >= 0 
    assert normal_ball[:,:,0].max() <= 1 
    
    # camera is pointing to the ball, assume that camera is othographic as it placing far-away from the ball
    I = np.array([1, 0, 0]) 
        
    ball_image = np.zeros_like(normal_ball)
    
    reflected_rays = get_reflection_vector_map(I[None,None], normal_ball)
    spherical_coords = cartesian_to_spherical(reflected_rays)
    
    theta_phi = spherical_coords[...,1:]
    
    # scale to [0, 1]
    # theta is in range [-pi, pi],
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi] 
    theta_phi[...,1] = theta_phi[...,1] / np.pi
    
    # mirror environment map because it from inside to outside
    theta_phi = 1.0 - theta_phi
    
    with torch.no_grad():
        # convert to torch to use grid_sample
        theta_phi = torch.from_numpy(theta_phi[None])
        env_map = torch.from_numpy(env_map[None]).permute(0,3,1,2)
        # grid sample use [-1,1] range
        grid = (theta_phi * 2.0) - 1.0
        ball_image = torch.nn.functional.grid_sample(env_map.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=True)
    return ball_image, normal_ball