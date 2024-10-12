
import torch
from DDIMInversion import DDIMInversion

class DDIMInversionForControlNet(DDIMInversion):

    def set_timesteps(self, target_timestep, num_inference_steps, device="cuda"):
        super().set_timesteps(target_timestep, num_inference_steps, device=device)
        #self.timesteps = torch.clamp(self.timesteps, min=0)

    def __call__(self, z, embedd, target_timestep, num_inference_steps, guidance_scale=1, device="cuda", controlnet_cond=None, cond_scale=1):
        self.set_timesteps(target_timestep, num_inference_steps, device=device)
        z_t = z.clone()
        with torch.no_grad():
            for timestep in self.timesteps:
                
                # forward the controlnet
                down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                    z_t,
                    timestep,
                    encoder_hidden_states=embedd,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    return_dict=False,
                )

                #eps = self.pipe.unet(z_t, timestep, embedd).sample

                # eps is a noise shape [1,4,64,64]
                
                eps = self.pipe.unet(
                    z_t,
                    timestep,
                    encoder_hidden_states=embedd,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample


                z_t = self.step_ddim_inv(z_t, eps, timestep)

        return z_t, eps