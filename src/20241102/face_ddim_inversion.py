from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from diffusers.utils import pt_to_pil
import json

import skimage 

import os
import torch
import numpy as np 
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from tqdm.auto import tqdm

SEED = 42
FILENAME = "00086.jpg"

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
    GUIDANCE_SCALE = 1.0
    NUM_INFERENCE = 500
    with open('/ist/ist-share/vision/relight/datasets/face/face60k/boy.json') as f:
        image_ids = json.load(f)

    # load model 
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,safety_checker=None)
    pipe = pipe.to("cuda")

    normal_scheduler = DDIMScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')



    for direction in ['left','right']:
        PROMPT_FROM = "face of a boy"
        PROMPT_TO = f"face of a boy with sunlight illuminate on the {direction}"
        embed_from = get_text_embeddings(pipe, PROMPT_FROM).to(pipe.unet.dtype)
        embed_to = get_text_embeddings(pipe, PROMPT_TO).to(pipe.unet.dtype)
        negative_embedding = get_text_embeddings(pipe, '').to(pipe.unet.dtype)
        for guidance_scale in [7.0]:
            output_dir = f"output/20241102/{guidance_scale:.1f}/{direction}/"
            os.makedirs(output_dir, exist_ok=True)
            print(output_dir)
            for image_id in tqdm(image_ids):
                dir_id = image_id // 1000 * 1000
                image_path = f"/ist/ist-share/vision/relight/datasets/face/face60k/images/{dir_id:05d}/{image_id:05d}.jpg"
                
                # load schuduler
                # read image 
                image = skimage.io.imread(image_path)
                image = skimage.transform.resize(image, (512,512))
                image = torch.from_numpy(image)
                image = image * 2.0 - 1.0
                image = image.permute(2,0,1)[None]

                # compute z0
                z0 = get_latent_from_image(pipe.vae, image.to('cuda'), generator=torch.Generator().manual_seed(SEED))

                # DDIM Inversion
                pipe.scheduler = inverse_scheduler
                ddim_args = {
                    "latents": z0,
                    "prompt_embeds": embed_from,
                    "negative_prompt_embeds": negative_embedding,
                    "guidance_scale": guidance_scale,
                    "return_dict": False,
                    "num_inference_steps": NUM_INFERENCE,
                    "generator": torch.Generator().manual_seed(SEED),        
                    "output_type": 'latent'
                }
                zt_noise, _ = pipe(**ddim_args)

                # DDIM Denoise
                pipe.scheduler = normal_scheduler
                pipe_args = {
                    "latents": zt_noise,
                    "negative_prompt_embeds": negative_embedding,
                    "prompt_embeds": embed_to,
                    "output_type": "pt",
                    "guidance_scale": guidance_scale,
                    "return_dict": False,
                    "num_inference_steps": NUM_INFERENCE,
                    "output_type": "pt",
                    "generator": torch.Generator().manual_seed(SEED)
                }
                pt_image, _ = pipe(**pipe_args)

                out_image = pt_image[0].permute(1,2,0).cpu().numpy()
            
                out_image = np.clip(out_image, 0, 1)
                out_image = skimage.img_as_ubyte(out_image)
                skimage.io.imsave(os.path.join(output_dir, f"{image_id:05d}.jpg") , out_image)


if __name__ == "__main__":
     main()