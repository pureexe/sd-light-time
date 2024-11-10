import torch
from diffusers import StableDiffusion3Pipeline
from attention_processor import RelightAttnProcessor2_0

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16, safety_checker=None)
pipe = pipe.to("cuda")
#pipe.transformer.set_attn_processor(RelightAttnProcessor2_0())

image = pipe(
    #"A capybara holding a sign that reads Hello World",
    "A photo of Lumine from Genshin impact",
    num_inference_steps=40,
    guidance_scale=4.5
).images[0]
image.save("src/20241104/lumine.png")