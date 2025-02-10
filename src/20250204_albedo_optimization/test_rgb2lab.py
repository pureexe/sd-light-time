import torch 
from PIL import Image 
import numpy as np 
import skimage

def rgb2lab(img):
    """
    Convert RGB to LAB
    """

    img = skimage.color.rgb2lab(img)

    # Normalize L (0-100) → (0,1) and A/B (-128,127) → (-1,1)
    img[:,:,0] = img[:,:,0] / 100
    img[:,:,1] = (img[:,:,1] + 128) / 255.0
    img[:,:,2] = (img[:,:,2] + 128) / 255.0

    # Convert range (0,1) → (-1,1)
    img = img * 2.0 - 1.0
    return img

def lab2rgb(img):
    is_torch = torch.is_tensor(img)

    if is_torch:
        device = img.device
        img = img.permute(1,2,0).cpu().numpy()

    # Convert from (-1,1) back to (0,1)
    img = (img + 1.0) / 2.0

    # Convert to LAB space L (0-100), A/B (-128,127)
    img[:,:,0] = np.clip(img[:,:,0], 0, 1) * 100  # L should be clipped
    img[:,:,1] = img[:,:,1] * 255 - 128
    img[:,:,2] = img[:,:,2] * 255 - 128

    # Convert LAB to RGB
    img = skimage.color.lab2rgb(img)

    if is_torch:
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).to(device)

    return img

def main():
    IMAGE_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10/dir_0_mip2.jpg"
    img = Image.open(IMAGE_PATH).convert("RGB")  # Ensure 3 channels
    img = rgb2lab(img)
    img = lab2rgb(img)
    img = skimage.img_as_ubyte(img)
    img = skimage.io.imsave("rgb2lab.png", img)


if __name__ == "__main__":
    main()