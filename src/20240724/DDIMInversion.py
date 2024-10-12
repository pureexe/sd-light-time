import numpy as np 
import torch
from diffusers import DDIMInverseScheduler

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

    def step_ddim_inv(self, z, eps, timestep):
        prev_timestep = timestep + self.target_timestep // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (z - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * eps
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return prev_sample

    def __call__(self, z, embedd, target_timestep, num_inference_steps, guidance_scale=1, device="cuda"):
        self.set_timesteps(target_timestep, num_inference_steps, device=device)
        z_t = z.clone()
        with torch.no_grad():
            for timestep in self.timesteps:
                eps = self.pipe.unet(z_t, timestep, embedd).sample
                z_t = self.step_ddim_inv(z_t, eps, timestep)

        return z_t, eps