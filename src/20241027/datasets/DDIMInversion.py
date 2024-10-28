import numpy as np 
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

class DDIMInversion():
    def __init__(self, pipe):
        # get params from DDIMScheduler
        self.pipe = pipe
        scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        self.alphas_cumprod = scheduler.alphas_cumprod
        self.initial_alpha_cumprod = scheduler.initial_alpha_cumprod

        print(self.pipe.scheduler.config.clip_sample)

    def set_timesteps(self, target_timestep, num_inference_steps, device="cuda"):
        self.target_timestep = target_timestep
        self.num_inference_steps = num_inference_steps

        step_ratio = self.target_timestep // self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round().copy().astype(np.int64) + 1

        timesteps = np.roll(timesteps, 1)
        timesteps[0] = int(timesteps[1] - step_ratio)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps = torch.clamp(self.timesteps, 0, self.target_timestep - 1)

    def step_ddim_inv(self, z, eps, timestep):
        prev_timestep = timestep + self.target_timestep // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (z - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * eps
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample

    def __call__(self, z, embedd, target_timestep, num_inference_steps, guidance_scale=1, negative_embedd=None, device="cuda", controlnet_cond=None, cond_scale=1):
        # check input 
        if guidance_scale > 1 and negative_embedd is None:
            raise ValueError("negative_embedd must be provided when guidance_scale > 1")
        
        #do_classifier_free_guidance = guidance_scale > 1
        # TODO: fully support inversion with guidance scale
        do_classifier_free_guidance = False
        
        prompt_embeds = torch.cat([negative_embedd, embedd], dim=0) if do_classifier_free_guidance else embedd

        self.set_timesteps(target_timestep, num_inference_steps, device=device)
        latents = z.clone()
        with torch.no_grad():
            for timestep in self.timesteps:

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.pipe, 'controlnet'):
                    down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                        latents,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None

                noise_pred = self.pipe.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                latents = self.step_ddim_inv(latents, noise_pred, timestep)

        return latents, noise_pred