import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import AutoencoderTiny
import os


with torch.no_grad():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")
    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipe(prompt=prompt).images[0]
    image.save()

os.system('nvidia-smi')