# make sure that invert and sample actually get back to the same value

import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
import numpy as np 
import os 

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def sample_step(
    pipe,
    prompt,
    t, # this will be t and t-1, so minimum t that can be passed is 1
    start_latents=None,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):  
    assert t > 0, "t must be greater than 0"
    # Encode prompt
    # force apply classifier free guidance here to avoid library bug
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, True, negative_prompt
    )
    if not do_classifier_free_guidance:
        _, text_embeddings = text_embeddings.chunk(2)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Normally we'd rely on the scheduler to handle the update step:
    # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Instead, let's do it ourselves:
    prev_t = t - 1  # t-1
    alpha_t = pipe.scheduler.alphas_cumprod[t]
    alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
    predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
    latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    return latents, noise_pred

## Inversion
@torch.no_grad()
def invert_step(
    pipe,
    prompt,
    t, # this will be t and t+1, so minimum t that can be passed is 0
    start_latents,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):
    # Encode prompt
    # force apply classifier free guidance here to avoid library bug
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, True, negative_prompt
    )
    if not do_classifier_free_guidance:
        _, text_embeddings = text_embeddings.chunk(2)
    

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual
    
    # # clone all input to avoid effect from in-place updating
    # latent_model_input1 = latent_model_input.clone()
    # latent_model_input2 = latent_model_input.clone()
    # text_embbedding1 = text_embeddings.clone()
    # text_embbedding2 = text_embeddings.clone()


    # # seed everything 
    # random.seed(42)
    # os.environ['PYTHONHASHSEED'] = str(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # # call unet 
    # noise_pred = pipe.unet(latent_model_input1, t, encoder_hidden_states=text_embbedding1).sample

    # # Re-seed everything again
    # random.seed(42)
    # os.environ['PYTHONHASHSEED'] = str(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # # call unet again
    # noise_pred2 = pipe.unet(latent_model_input2, t, encoder_hidden_states=text_embbedding2).sample

    # # print the max different 
    # print("NOISE PRED DIFF: ", (noise_pred - noise_pred2).abs().max())
    # exit()

    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    current_t = t  # t
    next_t = t+1  # min(999, t.item() + (1000//num_inference_steps)) # t+1
    alpha_t = pipe.scheduler.alphas_cumprod[current_t]
    alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

    # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
    latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
        1 - alpha_t_next
    ).sqrt() * noise_pred

    return latents, noise_pred

def main():
    MASTER_TYPE = torch.float32
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.inference_mode():
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=MASTER_TYPE).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(1000, device=device)
    
        #input_image = load_image("https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg", size=(512, 512))
        input_image = Image.open("input_dog.jpeg").resize((512, 512))

        input_image_prompt = "Photograph of a puppy on the grass"

        # Encode with VAE
        with torch.no_grad():
            latent = pipe.vae.encode((tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)* 2 - 1).to(MASTER_TYPE))
        latent = 0.18215 * latent.latent_dist.sample(generator)
        
        GUIDANCE_SCALE = 1.0

        latent1, noise1 =  invert_step(
            pipe,
            input_image_prompt,
            0, # this will be t and t+1, so minimum t that can be passed is 0
            latent,
            guidance_scale=GUIDANCE_SCALE,
            do_classifier_free_guidance=False
        )
        latent0, noise0 = sample_step(
            pipe,
            input_image_prompt,
            1, # this will be t and t-1, so minimum t that can be passed is 1
            start_latents=latent1,
            guidance_scale=GUIDANCE_SCALE,
            do_classifier_free_guidance=False
        )
        print("GUIDANCE_SCALE: ", GUIDANCE_SCALE)
        print("MAX DRIFT: ", (latent1 - latent0).abs().max())
        print("MEAN DRIFT: ", (latent1 - latent0).abs().mean())
        print("MAX NOISE DRIFT: ", (noise1 - noise0).abs().max())
        print("MEAN NOISE  DRIFT: ", (noise1 - noise0).abs().mean())
    


if __name__ == "__main__":
    main()


    """
    GUIDANCE_SCALE:  3.5
    MAX DRIFT:  tensor(0.0195, device='cuda:0', dtype=torch.float16)
    MEAN DRIFT:  tensor(0.0024, device='cuda:0', dtype=torch.float16)
    """

    """
    GUIDANCE_SCALE:  1.0
    MAX DRIFT:  tensor(0.0176, device='cuda:0', dtype=torch.float16)
    MEAN DRIFT:  tensor(0.0024, device='cuda:0', dtype=torch.float16)
    """

    """
    WITHOUT GUIDANCE_SCALE
    GUIDANCE_SCALE:  1.0
    MAX DRIFT:  tensor(0.0176, device='cuda:0', dtype=torch.float16)
    MEAN DRIFT:  tensor(0.0024, device='cuda:0', dtype=torch.float16)
    """

    """
    WITHOUT GUIDANCE_SCALE FLOAT 32
    GUIDANCE_SCALE:  1.0
    MAX DRIFT:  tensor(0.0173, device='cuda:0')
    MEAN DRIFT:  tensor(0.0024, device='cuda:0')
    """