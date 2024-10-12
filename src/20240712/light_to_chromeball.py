import pyshtools
import numpy as np 
from LineNotify import notify
from sh_utils import compute_background
from PIL import Image
#01000/01237

#@notify
def main():
    fname = "01000/01237"
    sh_coeff = np.load(f"datasets/face/face2000_single/light/{fname}_light.npy")
    sh_coeff = sh_coeff.transpose(1, 0)
    image = compute_background(60, sh_coeff, lmax=2, image_width=256, show_entire_env_map=True)
    print(image.shape)
    print(image.max())
    print(image.min())


if __name__ == "__main__":
    main()

