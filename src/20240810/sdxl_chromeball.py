# create a chromeball inpainting to see the result from classic sd 

import torch
from diffusers import StableDiffusionControlNetInpaintPipeline , StableDiffusionXLPipeline, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL import Image
from transformers import pipeline as transformers_pipeline

import os
from constants import OUTPUT_CHROMEBALL_DIR, OUTPUT_CHROMEBALL_SDXL_DIR

import numpy as np
import skimage 

PROMPT = "a perfect mirrored reflective chrome ball sphere"
NEGATIVE_PROMPT = "matte, diffuse, flat, dull, disco"
#MODE = "sdxl"
MODE = "sd"

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


def apply_circle_mask(image, mask):
    """
    Processes two PIL images: an original image and a binary mask.

    Returns:
        A PIL image with modified pixels based on the mask.
    """

    # Ensure both images have the same size
    if image.size != mask.size:
        raise ValueError("Image and mask sizes do not match.")

    # Convert images to NumPy arrays for efficient processing
    image_data = np.array(image)
    mask_data = np.array(mask)

    # Apply mask to image data
    image_data[mask_data == 255] = 255

    # Convert back to PIL image
    result_image = Image.fromarray(image_data)

    return result_image


def get_pil_mask_image(num_pixel=512, circle_diameter=256):

    mask_image = create_circle_image(num_pixel, circle_diameter)
    mask_image = skimage.img_as_ubyte(mask_image)
    mask_image = Image.fromarray(mask_image).convert("L")
    return mask_image

def main():
    circle_image = get_pil_mask_image(circle_diameter=128)
    mask_image = get_pil_mask_image(circle_diameter=138)

    INPUT_DIR = "/data/pakkapon/datasets/unsplash-lite/train/images"
    files = sorted(os.listdir(INPUT_DIR))

    depth_estimator = transformers_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-base-hf", device="cuda")
    

    with torch.no_grad():
        if MODE == "sdxl":
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", torch_dtype=torch.float16)
            #sdxl_pipeclass = StableDiffusionXLPipeline  
            sdxl_pipeclass = StableDiffusionXLControlNetPipeline
            #sdxl_pipeclass = StableDiffusionXLControlNetInpaintPipeline
            pipe = sdxl_pipeclass.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                variant="fp16",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
            ).to("cuda")
            #vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            output_dir = OUTPUT_CHROMEBALL_SDXL_DIR
        else:
            output_dir = OUTPUT_CHROMEBALL_DIR
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
            ).to("cuda")
        for idx, fname in enumerate(files[100:]):
            print(f"Processing {fname} ({idx+1}/{len(files)})")
            init_image = Image.open(os.path.join(INPUT_DIR, fname)).convert("RGB")
            depth_image = depth_estimator(init_image)['depth']

            depth_masked_image = apply_circle_mask(depth_image, circle_image)   


            result = pipe(
                prompt=PROMPT, 
                negative_prompt=NEGATIVE_PROMPT,
                image=init_image, 
                mask_image=mask_image, 
                control_image=depth_masked_image,
                #image=depth_masked_image,
                num_inference_steps=30,
                guidance_scale=5.0
            ).images[0]
            os.makedirs(output_dir, exist_ok=True)
            result.save(os.path.join(output_dir, f"{fname}"))







if __name__ == '__main__':
    main()