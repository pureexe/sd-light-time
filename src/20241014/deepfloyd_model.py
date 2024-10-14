from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()


prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

def callback_fn(i, t, latents):
    # print latents shape
    print("latents: " ,latents.shape)


image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", callback=callback_fn).images

# save intermediate image
pil_image = pt_to_pil(image)
pil_image[0].save("./if_stage_I.png")

super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
super_res_1_pipe.enable_model_cpu_offload()

image = super_res_1_pipe(
    image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
).images

# save intermediate image
pil_image = pt_to_pil(image)
pil_image[0].save("./if_stage_II.png")

