import pyshtools
import skimage
import numpy as np
import torch
import os


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

def generate_sphere_map_theta(image_size = 256, offset=np.pi/2):
    """
    Generate a tuple of numpy arrays representing the latitude and longitude
    of a spherical map in a circular shape.

    Parameters:
        image_size (int): The size (width and height) of the square image.

    Returns:
        tuple: A tuple containing two numpy arrays (latitude, longitude).
    """
    # Create linear ranges for latitude and longitude
    latitude = np.linspace(np.pi / 2, -np.pi / 2, image_size)  # [pi/2, -pi/2]
    longitude = np.linspace(0 + offset, np.pi + offset, image_size)  # [0, 2*pi]
    
    # Create 2D grid for latitude and longitude
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    
    # Create a mask to remove points outside the circle
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + yv**2)
    mask = radius <= 1  # Valid points within the circle
    
    # Apply the mask
    lat_grid[~mask] = 0
    lon_grid[~mask] = 0
    
    return lat_grid, lon_grid

def cartesian_to_spherical(vectors):
    """
    Converts unit vectors to spherical coordinates (theta, phi).

    Parameters:
    vectors (numpy.ndarray): Input array of shape (..., 3), representing unit vectors.

    Returns:
    tuple: A tuple containing two arrays:
        - theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
        - phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].
    """
    # Ensure input is a numpy array
    vectors = np.asarray(vectors)

    # Validate shape
    if vectors.shape[-1] != 3:
        raise ValueError("Input must have shape (..., 3).")

    # Extract components of the vectors
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]

    # Calculate theta (latitude angle)
    theta = np.arcsin(y)  # arcsin gives range [-pi/2, pi/2]

    # Calculate phi (longitude angle)
    phi = np.arctan2(x, z)  # atan2 accounts for correct quadrant
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # Normalize phi to range [0, 2*pi]

    return theta, phi

def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            for k in range(j, 0, -1):
                sh_coeff[i, 1, j, k] = flatted_coeff[i, c]
                c +=1
            for k in range(j+1):
                sh_coeff[i, 0, j, k] = flatted_coeff[i, c]
                c += 1
    return sh_coeff


def main():
    COEFF_DIRS = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/shcoeffs"
    SCENES = ["14n_copyroom1"]
    OUTPUT_DIR = "output/chromeball_ldr_order2"

    # get normal ball
    BALL_SIZE = 128
    normal_image, mask = get_ideal_normal_ball(BALL_SIZE)
    theta, phi = cartesian_to_spherical(normal_image)

    #theta, phi = generate_sphere_map_theta(128, np.pi/2)

    for scene in SCENES:
        output_dir = os.path.join(OUTPUT_DIR, scene)
        coeff_dir = os.path.join(COEFF_DIRS, scene)
        os.makedirs(output_dir, exist_ok=True)
        all_images = [f.replace(".npy","")  for f in sorted(os.listdir(coeff_dir))]
        for fname in all_images:
            print(coeff_dir, '/', fname+".npy")
            shcoeffs = np.load(os.path.join(coeff_dir, fname+".npy"))
            shcoeffs = unfold_sh_coeff(shcoeffs)
            shading = sample_envmap_from_sh(shcoeffs, 2, theta, phi)
            shading = np.clip(shading,0,1)
            shading = skimage.img_as_ubyte(shading)
            skimage.io.imsave(os.path.join(output_dir,fname+".png"), shading)

if __name__ == "__main__":
    main()