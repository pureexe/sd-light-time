from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from diffusers.utils import pt_to_pil

import skimage 

import torch
import numpy as np 
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline


SEED = 90
FILENAME = f"face_source_g7.5_seed{SEED}.jpg"

@torch.inference_mode()
def get_text_embeddings(pipe, text):
    if isinstance(text, str):
            text = [text]

    tokens = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(pipe.text_encoder.device)
    
    return pipe.text_encoder(tokens).last_hidden_state

@torch.inference_mode()
def get_latent_from_image(vae, image, generator=None):
    """_summary_

    Args:
        vae (_type_): VAE Autoencoder class
        image (_type_): image in format [-1,1]

    Returns:
        _type_: _description_
    """
  
    latents =  vae.encode(image.to(vae.dtype)).latent_dist.sample(generator=generator)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(vae.dtype)
    return latents

@torch.inference_mode()
def get_image_from_latent(vae, latents, generator=None):
    """_summary_

    Args:
        vae (_type_): VAE Autoencoder class
        image (_type_): image in format [-1,1]

    Returns:
        _type_: _description_
    """

    image = vae.decode(latents / vae.config.scaling_factor).sample
    image = image.to(vae.dtype)
    return latents

@torch.inference_mode()
def main():
    for SEED in range(100):

        GUIDANCE_SCALE = 7.5
        NUM_INFERENCE = 500
        # load model 
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,safety_checker=None)
        pipe = pipe.to("cuda")
    
        pipe_args = {
            "prompt": "face of a boy with sunlight illuminate on the right",
            "output_type": "pt",
            "guidance_scale": GUIDANCE_SCALE,
            "return_dict": False,
            "num_inference_steps": NUM_INFERENCE,
            "output_type": "pt",
            "generator": torch.Generator().manual_seed(SEED)
        }
        pt_image, _ = pipe(**pipe_args)
        out_image = pt_image[0].permute(1,2,0).cpu().numpy()
    
        out_image = np.clip(out_image, 0, 1)
        out_image = skimage.img_as_ubyte(out_image)
        skimage.io.imsave(f"src/20241101/output/{FILENAME}", out_image)


if __name__ == "__main__":
     main()