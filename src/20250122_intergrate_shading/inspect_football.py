import numpy as np
import skimage 
from sh_utils import from_x_left_to_z_up
def main():
    normal = np.load("output/efficient_rendering/ball1.npy") #512,512,3 (range -1,1)
    # Compute the vector norm along the last axis
    norm = np.linalg.norm(normal, axis=2, keepdims=True)

    # Avoid division by zero
    norm[norm == 0] = 1
    normal = normal / norm
    print(normal.shape)
    normal = from_x_left_to_z_up(normal)
    normal = (normal + 1.0) / 2.0
    print(normal.min())
    print(normal.max())
    img = skimage.img_as_ubyte(normal)
    print("IMAGE SHAPE:",img.shape)
    skimage.io.imsave("output/efficient_rendering/ball_normal1_convert_axis_v4.png", img)

if __name__ == "__main__":
    main()