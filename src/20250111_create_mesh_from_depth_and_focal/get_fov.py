import numpy as np 

#FOV_PX = 810.4429321289062
FOV_PX = 618.0386719675123
IMAGE_WIDTH = 512

fov_rad = 2 * np.arctan2(IMAGE_WIDTH, 2*FOV_PX)
fov_deg = fov_rad / np.pi * 180
print(fov_deg)