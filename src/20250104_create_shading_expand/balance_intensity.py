import numpy as np
import skimage 

import os 
from tqdm.auto import tqdm

SOURCE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_copyroom10_left/control_shading_from_ldr27coeff/14n_copyroom10"
SOURCE_IMAGE = "dir_0_mip2.png"
TARGET_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_copyroom10_left/control_shading_from_ldr27coeff_balance/14n_copyroom10"

def convert_to_grayscale_and_average_intensity(image):
    """
    Convert an RGB image to grayscale and calculate the average intensity.

    Parameters:
        image (numpy.ndarray): A NumPy array of shape [H, W, 3] representing the RGB image.

    Returns:
        grayscale_image (numpy.ndarray): A NumPy array of shape [H, W] representing the grayscale image.
        average_intensity (float): The average intensity of the grayscale image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape [H, W, 3].")

    # Convert to grayscale using the luminosity method
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Calculate the average intensity
    average_intensity = np.mean(grayscale_image)

    return grayscale_image, average_intensity

def adjust_image_to_average_intensity(target_intensity, image):
    """
    Adjust the RGB image to have the specified average intensity.

    Parameters:
        target_intensity (float): The desired average intensity.
        image (numpy.ndarray): A NumPy array of shape [H, W, 3] representing the RGB image.

    Returns:
        adjusted_image (numpy.ndarray): A NumPy array of shape [H, W, 3] with the adjusted intensity.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape [H, W, 3].")

    # Convert to grayscale to calculate the current average intensity
    grayscale_image, current_intensity = convert_to_grayscale_and_average_intensity(image)

    # Calculate the scaling factor
    scaling_factor = target_intensity / (current_intensity + 1e-8)  # Avoid division by zero

    # Adjust the image
    adjusted_image = image * scaling_factor

    return adjusted_image

# Example usage
if __name__ == "__main__":
    # Load the source image
    source_image = skimage.io.imread(os.path.join(SOURCE_DIR,SOURCE_IMAGE))
    source_image = skimage.img_as_float32(source_image)

    # get the average intensity of the source image
    _, source_intensity = convert_to_grayscale_and_average_intensity(source_image)
    print("Source image average intensity:", source_intensity)

    os.makedirs(TARGET_DIR, exist_ok=True)

    files = os.listdir(SOURCE_DIR)
    for filename in files:
        source_image = skimage.io.imread(os.path.join(SOURCE_DIR,filename))
        source_image = skimage.img_as_float32(source_image)

        # Adjust the image to have an average intensity of source_intensity
        adjusted_image = adjust_image_to_average_intensity(source_intensity, source_image)

        _, target_intensity = convert_to_grayscale_and_average_intensity(adjusted_image)
        print("Target image average intensity:", target_intensity)

        adjusted_image = np.clip(adjusted_image,0 ,1)

        adjusted_image = skimage.img_as_ubyte(adjusted_image)

        # Save the adjusted image
        skimage.io.imsave(os.path.join(TARGET_DIR,filename), adjusted_image)
    