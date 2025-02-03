import numpy as np 
import skimage 

def main():
    image = skimage.io.imread("data/coordinates_z_up.png")[...,:3]
    image = skimage.img_as_float(image)
    image = np.roll(image, 512, axis=1)
    image = image[:,::-1]
    image = skimage.img_as_ubyte(image)
    image = skimage.io.imsave("data/coordinates_z_up_pi_to_-pi.png", image)


if __name__ == "__main__":
    main()