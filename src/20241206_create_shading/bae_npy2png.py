import numpy as np 
import skimage

def main():
    bae = np.load("normal_bae.npy")
    # convert to [0,1]
    bae = (bae + 1.0) / 2.0
    bae = skimage.img_as_ubyte(bae)
    skimage.io.imsave("normal_bae.png", bae)
    exit()

if __name__ == "__main__":
    main()