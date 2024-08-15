from diffusers import StableDiffusionPipeline
import torch
import os 

GUIDANCE_SCALE = 7.0


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    OUTPUT_DIR = "output/20240811/check_shoe100_seed42_cat/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompt = "a photo of cat sitting on the field at the morning"
    for image_id in range(1):
        image = pipe(
            prompt, 
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=50,
            generator=torch.Generator().manual_seed(42)
        ).images[0] 
            
        image.save(OUTPUT_DIR + f"{image_id:05d}.png")

if __name__ == "__main__":
    main()