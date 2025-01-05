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

def main():
    image = skimage.io.imread("coordinates.png")
    image = skimage.img_as_float(image)[...,:3]
    shcoeffs = get_shcoeff(image)
    theta = np.array([
        np.pi/2,
        0,
        0,
        0,
        0,
        -np.pi/2,
    ])
    phi = np.array([
        np.pi,
        0,
        np.pi/2,
        np.pi,
        np.pi * 3 / 2,
        np.pi,
    ])
    output_image = sample_envmap_from_sh(shcoeffs, 100, theta, phi) #shape 6,3
    output_image = np.clip(output_image, 0, 1)
    for i in range(theta.shape[0]):
        color = np.ones((256,256,3))
        color[:,:] = output_image[i]
        color = skimage.img_as_ubyte(color)
        skimage.io.imsave(f"output/color6/color_{i:02d}.png", color)

if __name__ == "__main__":
    main()