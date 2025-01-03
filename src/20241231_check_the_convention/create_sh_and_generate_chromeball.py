import pyshtools
import skimage
import numpy as np
import torch

def get_shcoeff(image, Lmax=100):
    """
    @param image: image in HWC @param 1max: maximum of sh
    """
    output_coeff = []
    for c_id in range(image.shape[-1]):
        # Create a SHGrid object from the image
        grid = pyshtools.SHGrid.from_array(image[:,:,c_id], grid='GLQ')
        # Compute the spherical harmonic coefficients
        coeffs = grid.expand(normalization='4pi', csphase=1, lmax_calc=Lmax)
        coeffs = coeffs.to_array()
        output_coeff.append(coeffs[None])
    
    output_coeff = np.concatenate(output_coeff,axis=0)
    return output_coeff

def compute_background(
        sh, lmax, image_width=512
    ):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh


    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
        phi = np.linspace(0, np.pi * 2, 2*image_width)

        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    output_image = np.clip(output_image, 0.0 ,1.0)
    return output_image

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def sample_envmap_from_sh(shcoeff, lmax, theta, phi):
    """
    Sample envmap from sh 
    """
    assert shcoeff.shape[0] == 3 # make sure that it a 3 channel input
    output = []
    for ch in (range(3)):
        coeffs = pyshtools.SHCoeffs.from_array(shcoeff[ch], lmax=lmax, normalization='4pi', csphase=1)
        image = coeffs.expand(grid="GLQ", lat=theta, lon=phi, lmax_calc=lmax, degrees=False)
        output.append(image[...,None])
    output = np.concatenate(output, axis=-1)
    return output

def cartesian_to_spherical_grid(xyz_grid):
    """
    Convert a grid of Cartesian coordinates to spherical coordinates.

    Parameters:
        xyz_grid (ndarray): Array of shape [H, W, 3] representing Cartesian coordinates.

    Returns:
        tuple: Arrays (theta, phi) of shape [H, W], where
               theta is the colatitude (0 to pi),
               phi is the longitude (-pi to pi).
    """
    # Extract x, y, z components
    x = xyz_grid[..., 0]
    y = xyz_grid[..., 1]
    z = xyz_grid[..., 2]

    # Compute the spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)  # Radius (not returned, but could be useful)
    theta = np.arccos(z / np.clip(r, a_min=1e-8, a_max=None))  # Colatitude (0 to pi)
    phi = np.arctan2(y, x)  # Longitude (-pi to pi)

    return theta, phi


def main():
    image = skimage.io.imread("coordinates.png")[:,:,:3]
    image = skimage.img_as_float(image)
    shcoeffs = get_shcoeff(image)

    # get normal ball
    normal_image, _ = get_ideal_normal_ball(128)
    normal_image_flip = normal_image.copy()
    normal_image_flip[:,:,2] = -normal_image_flip[:,:,2]
    normal_image = np.concatenate([normal_image,normal_image_flip],axis=1)
    normal_map = normal_image
    normal_map = torch.from_numpy(normal_map)
    normal_map = normal_map.permute(2,0,1)[None]

    theta, phi = cartesian_to_spherical_grid(normal_map[0].permute(1,2,0).numpy())
    THETA_MIN = 0
    THETA_MAX = np.pi
    PHI_MIN = -np.pi
    PHI_MAX = np.pi

    shading = sample_envmap_from_sh(shcoeffs, 100, theta, phi)
    shading = np.clip(shading, 0, 1)

    normal_map_rgb = normal_map[0].permute(1,2,0).numpy()
    normal_map_rgb = (normal_map_rgb + 1.0 )/ 2.0
    normal_map_rgb = skimage.img_as_ubyte(normal_map_rgb)
    skimage.io.imsave("output_normal.png", normal_map_rgb)

    n_theta = (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)
    n_phi = (phi - PHI_MIN) / (PHI_MAX - PHI_MIN)
    n_ones = np.ones_like(n_phi) / 2.0

    
    n_theta_phi = np.concatenate([n_theta[...,None], n_phi[...,None], n_ones[...,None]], axis=-1)

    n_theta_phi = skimage.img_as_ubyte(n_theta_phi)

    skimage.io.imsave("output_theta_phi.png", n_theta_phi)

    output_image = skimage.img_as_ubyte(shading)
    skimage.io.imsave("output_generatedchromeball.png", output_image)

if __name__ == "__main__":
    main()