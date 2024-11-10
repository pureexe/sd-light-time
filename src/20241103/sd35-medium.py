import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    safety_checker=None
)
pipe = pipe.to("cuda")
print(pipe.transformer)
exit()
