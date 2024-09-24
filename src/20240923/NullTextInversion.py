import numpy as np 
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

class NullTextnversion():
    def __init__(self, pipe):
        # get params from DDIMScheduler
        self.pipe = pipe
        self.normal_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config) 

    @torch.inference_mode()
    def get_text_embeddings(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        ).input_ids.to(self.pipe.text_encoder.device)
        return self.pipe.text_encoder(tokens).last_hidden_state
    
    def inverse_mode(self):
        self.pipe.scheduler = self.inverse_scheduler

    def normal_mode(self):
        self.pipe.scheduler = self.normal_scheduler

    def __call__(self, z, embedd, num_inference_steps, num_null_steps, negative_embedd=None, guidance_scale=1, device="cuda", controlnet_cond=None, cond_scale=1):
        
        if negative_embedd is None:
            negative_embedd = self.get_text_embeddings([""])

        do_classifier_free_guidance = guidance_scale > 1
        prompt_embeds = torch.cat([negative_embedd, embedd], dim=0) if do_classifier_free_guidance else embedd

        #self.set_timesteps(target_timestep, num_inference_steps, device=device)
        
        # DDIMInversion 
        self.inverse_mode()
        
        ddim_args = {
            "prompt_embeds": text_embbeding,
            "guidance_scale": 1.0,
        }
        
        self.pipe()

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