import pyshtools
import skimage
import numpy as np

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

def main():
    image = skimage.io.imread("coordinates.png")
    image = skimage.img_as_float(image)[...,:3]
    shcoeffs = get_shcoeff(image)

    theta, phi = generate_sphere_map_theta(256, np.pi/2)
    print(phi[1])
    exit()
    exit()
    output_image = sample_envmap_from_sh(shcoeffs, 100, theta, phi) #shape 6,3
    output_image = np.clip(output_image, 0, 1)
    output_image = skimage.img_as_ubyte(output_image)
    skimage.io.imsave(f"output/latlong_ball/front.png", output_image)

    theta, phi = generate_sphere_map_theta(256, 3*np.pi/2)
    output_image = sample_envmap_from_sh(shcoeffs, 100, theta, phi) #shape 6,3
    output_image = np.clip(output_image, 0, 1)
    output_image = skimage.img_as_ubyte(output_image)
    skimage.io.imsave(f"output/latlong_ball/back.png", output_image)



if __name__ == "__main__":
    main()