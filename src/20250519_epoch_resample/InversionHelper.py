import torch
import numpy as np 
from diffusers import DDIMInverseScheduler, DDIMScheduler

def get_latent_from_image(vae, image, generator=None):
    """_summary_

    Args:
        vae (_type_): VAE Autoencoder class
        image (_type_): image in format [-1,1]

    Returns:
        _type_: _description_
    """
    latents =  vae.encode(image).latent_dist.sample(generator=generator)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(vae.dtype)
    return latents

def get_ddim_latents(
        pipe,
        image,
        text_embbeding,
        num_inference_steps,
        generator = None,
        controlnet_image=None,
        guidance_scale=1.0,
        interrupt_index=None,
        controlnet_conditioning_scale=1.0,
    ):
    scheduler_config = pipe.scheduler.config

    normal_scheduler = DDIMScheduler.from_config(scheduler_config, subfolder='scheduler')
    inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config, subfolder='scheduler')

    pipe.scheduler = inverse_scheduler

    if hasattr(pipe, 'vae'):
        z0_noise = get_latent_from_image(pipe.vae, image)

    # do ddim inverse to noise 
    ddim_latents = []
    ddim_timesteps = []
    
    ddim_args = {
        "prompt_embeds": text_embbeding,
        "guidance_scale": guidance_scale,
        "return_dict": False,
        "num_inference_steps": num_inference_steps,
        "generator": generator,        
    }


    def callback_ddim(pipe, step_index, timestep, callback_kwargs):
        ddim_timesteps.append(timestep)
        ddim_latents.append(callback_kwargs['latents'])
        if interrupt_index is not None and step_index >= interrupt_index:
            pipe._interrupt = True
            return callback_kwargs
        return callback_kwargs

    ddim_args['latents'] = z0_noise
    ddim_args['output_type'] = 'latent'
    ddim_args["callback_on_step_end"] = callback_ddim

    if controlnet_image is not None:
        ddim_args['image'] = controlnet_image
        ddim_args['controlnet_conditioning_scale'] = controlnet_conditioning_scale

    zt_noise, _ = pipe(**ddim_args)

    pipe.scheduler = normal_scheduler
    return ddim_latents, ddim_timesteps

