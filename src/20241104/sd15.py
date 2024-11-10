import torch
from diffusers import StableDiffusionPipeline
from attention_processor import RelightAttnProcessor2_0, RelightSD15AttnProcessor2_0

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.unet.set_attn_processor(RelightSD15AttnProcessor2_0())

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("src/20241104/capybara_sd15.png")