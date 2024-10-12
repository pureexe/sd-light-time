# create a chromeball inpainting to see the result from classic sd 

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

import os
from constants import OUTPUT_CHROMEBALL_DIR

import numpy as np
import skimage 

PROMPT = "a perfect mirrored reflective chrome ball sphere"
NEGATIVE_PROMPT = "matte, diffuse, flat, dull, disco"

def create_circle_image(num_pixel=512, circle_diameter=256):
    # Initialize the array with zeros
    image = np.zeros((num_pixel, num_pixel), dtype=np.float32)
    
    # Calculate the center of the image
    center = (num_pixel // 2, num_pixel // 2)
    
    # Calculate the radius of the circle
    radius = circle_diameter / 2
    
    # Create a grid of coordinates
    y, x = np.ogrid[:num_pixel, :num_pixel]
    
    # Calculate the distance from the center
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    
    # Set the pixels inside the circle to 1
    mask = dist_from_center <= radius
    image[mask] = 1
    
    return image

def main():
    mask_image = create_circle_image()
    mask_image = skimage.img_as_ubyte(mask_image)
    mask_image = Image.fromarray(mask_image).convert("L")

    INPUT_DIR = "/data/pakkapon/datasets/unsplash-lite/train/images"
    files = os.listdir(INPUT_DIR)
    os.makedirs(OUTPUT_CHROMEBALL_DIR, exist_ok=True)

    with torch.no_grad():
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        for idx, fname in enumerate(files[:100]):
            print(f"Processing {fname} ({idx+1}/{len(files)})")
            init_image = Image.open(os.path.join(INPUT_DIR, fname)).convert("RGB")
            result = pipe(
                prompt=PROMPT, 
                negative_prompt=NEGATIVE_PROMPT,
                image=init_image, 
                mask_image=mask_image, 
                num_inference_steps=30
            ).images[0]
            result.save(os.path.join(OUTPUT_CHROMEBALL_DIR, f"{fname}"))







if __name__ == '__main__':
    main()