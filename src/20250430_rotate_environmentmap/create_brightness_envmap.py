import skimage
import numpy as np
import ezexr 

def main():
    image = skimage.io.imread("brightness_bright_1024.png")
    image = skimage.img_as_float(image)
    image = image * 1024
    # roll the image by half
    H, W, C = image.shape
    image = np.roll(image, W // 2, axis=1)
    # save the image
    ezexr.imwrite("brightness_bright_1024.exr", image)

if __name__ == "__main__":
    main()