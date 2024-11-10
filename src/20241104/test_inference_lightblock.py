import torch
from diffusers import StableDiffusion3Pipeline
from lightembedblock_sd3 import set_light_direction, add_light_block

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# register light condition
add_light_block(pipe, 27)

# apply light condition
set_light_direction(pipe, torch.ones((1,27)).to("cuda"), is_apply_cfg=True)
    
image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
    generator = torch.Generator().manual_seed(42)
).images[0]
image.save("src/20241104/capybara_kightdirection.png")