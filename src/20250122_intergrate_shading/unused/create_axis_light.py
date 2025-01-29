from PIL import Image
import numpy as np

# Image dimensions
width = 512
height = 256

# Create a black image
image = np.zeros((height, width), dtype=np.uint8)

"""
# Convert angles to pixel range
#angle_start = -22.5
#angle_end = 22.5

angle_start = 247.5
angle_end = 292.5

# Map angles to pixel indices
start_pixel = int((angle_start + 360) / 360 * width) % width
end_pixel = int((angle_end + 360) / 360 * width) % width

# Set pixels to white in the specified range
if start_pixel < end_pixel:
    image[:, start_pixel:end_pixel] = 255
else:  # Handle the wrap-around case
    image[:, start_pixel:] = 255
    image[:, :end_pixel] = 255

"""

# Convert angles to pixel range
angle_start = -45
angle_end = -90

# Map angles to pixel indices
start_pixel = int((90 - angle_start) / 180 * height)
end_pixel = int((90 - angle_end) / 180 * height)

# Set pixels to white in the specified range
if start_pixel < end_pixel:
    image[start_pixel:end_pixel, :] = 255
else:  # Handle the wrap-around case
    image[start_pixel:, :] = 255
    image[:end_pixel, :] = 255

# Create and save the image

# Create and save the image
output_image = Image.fromarray(image)
output_image.save("data/axis_light/z_minus.png")
print("Image saved ")