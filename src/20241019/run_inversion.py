import os
from pipeline import PureIFPipeline
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline, IFSuperResolutionPipeline
from diffusers.utils import pt_to_pil
#from transformers.utils import FrozenDict

import torch 
import torchvision

MASTER_TYPE = torch.float16
OUTPUT_DIR = "inversion_explore"
IMAGE_PATH = "src/20241019/dog_64.png"
SEED = 42

def main():
    print("READ IMAGE FILE")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),  # Resize to 64x64
        torchvision.transforms.ConvertImageDtype(torch.float),  # Convert to float
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])
    image = torchvision.io.read_image(IMAGE_PATH)
    image = transform(image)
    image = image.unsqueeze(0).to('cuda').half()

    print("LOADING PIPELINE...")
    pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16, safety_checker=None)    
    pipe = pipe.to('cuda')
    print("LOADING SCHEDULER...")
    scheduler_config = {k: v for k, v in pipe.scheduler.config.items() if k != "variance_type"}
    scheduler_config["variance_type"] = "fixed_small"
    normal_scheduler = DDIMScheduler.from_config(scheduler_config)
    inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    original_scheduler  = pipe.scheduler

    prompt = 'a photorealistic image'
    #prompt = 'a photo of a dog'
    #prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
    prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
    
    intermediate_images = []

    # inversion step 
    pipe.scheduler = inverse_scheduler


    def callback(i, t, latents):
        intermediate_images.append(latents)
    
    pipe.set_initial_image(image)

    z_t = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt", generator=torch.Generator().manual_seed(SEED),
        num_inference_steps=999,
        callback=callback,
        guidance_scale=1.0,
    ).images

    # save output image 
    pil_image = pt_to_pil(z_t)
    os.makedirs(f"src/20241019/output/{OUTPUT_DIR}", exist_ok=True)
    os.makedirs(f"src/20241019/output/{OUTPUT_DIR}/inversion", exist_ok=True)
    pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/noise.png")

    # save intermediate image
    for i, img in enumerate(intermediate_images):
        pil_image = pt_to_pil(img)
        pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/inversion/{i:03d}.png")

    # forward step
    pipe.scheduler = normal_scheduler
    #pipe.scheduler = original_scheduler
    #pipe.set_initial_image(None)
    pipe.set_initial_image(z_t)
    intermediate_images = []
    z_0 = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt", generator=torch.Generator().manual_seed(SEED),
        num_inference_steps=999,
        callback=callback,
        guidance_scale=1.0
    ).images

    
    # save output image
    print('save output image')
    pil_image = pt_to_pil(z_0)
    os.makedirs(f"src/20241019/output/{OUTPUT_DIR}/reconstruction", exist_ok=True)
    pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/reconstruction.png")

    # save intermediate image
    for i, img in enumerate(intermediate_images):
        pil_image = pt_to_pil(img)
        pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/reconstruction/{i:03d}.png")


    # load upscale model 
    super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    ).to('cuda')
    
    image = super_res_1_pipe(
        image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
    ).images
    pil_image = pt_to_pil(image)
    pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/reconstruction_upsized.png")
    

    



    
    #super_res_1_pipe.enable_model_cpu_offload()

    # prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
    # print("COMPUTING PROMPT EMBEDDINGS...")
    # prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
    # print("INFERENCING PIPELINE...")
    # for seed in range(100):
    #     image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", generator=torch.Generator().manual_seed(seed)).images
    #     #print min max of image
    #     print("image min: ", image.min())
    #     print("image max: ", image.max())
    #     exit()
    #     pil_image = pt_to_pil(image)
    #     pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/64/seed_{seed:03d}.png")
    #     image = super_res_1_pipe(
    #         image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
    #     ).images

    #     # save intermediate image
    #     pil_image = pt_to_pil(image)
    #     pil_image[0].save(f"src/20241019/output/{OUTPUT_DIR}/256/seed_{seed:03d}.png")
        

if __name__ == "__main__":
    main()

    