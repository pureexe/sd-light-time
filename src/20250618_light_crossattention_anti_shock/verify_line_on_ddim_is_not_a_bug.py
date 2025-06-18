import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image 
import numpy as np
MASTER_TYPE = torch.float16

def get_ddim_latents(pipe, sharp_latents, inverse_scheduler, normal_scheduler, prompt_embeds=None):

    device = sharp_latents.device
    
    # swap the scheduler to the inverse 
    pipe.scheduler = inverse_scheduler

    ddim_latents = []
    ddim_timesteps = []

    def callback_ddim_on_step_end(p, i, t, callback_kwargs):
        ddim_timesteps.append(t)
        ddim_latents.append(callback_kwargs['latents'])
        return callback_kwargs
            
    pipe_args = {
        "latents": sharp_latents,  # [B, C*4, H, W]
        "prompt_embeds": prompt_embeds,  # [B, C*3, H, W]
        "negative_prompt_embeds": prompt_embeds,  # [B, C*3, H, W]
        "output_type": "pt",
        "guidance_scale": 1.0,
        "return_dict": False,
        "num_inference_steps": 500,
        "generator": torch.Generator().manual_seed(42),
        "callback_on_step_end": callback_ddim_on_step_end,
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds", "negative_prompt_embeds"],
    }
    
    zt_noise, _ = pipe(**pipe_args)

    # swap the scheduler back 
    pipe.scheduler = normal_scheduler
    return ddim_latents[-1]

def generate_images(pipe, noisy_latents, prompt_embeds):
    device = noisy_latents.device
    pipe_args = {
        "latents": noisy_latents,  # [B, C*4, H, W]
        "prompt_embeds": prompt_embeds,  # [B, C*3, H, W]
        "negative_prompt_embeds": prompt_embeds,  # [B, C*3, H, W]
        "output_type": "pt",
        "guidance_scale": 1.0,
        "return_dict": False,
        "num_inference_steps": 500,
        "generator": torch.Generator().manual_seed(42),
    }

    pt_image, _ = pipe(**pipe_args)
    return pt_image

@torch.inference_mode()
def main():
    pipe =  StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        torch_dtype=MASTER_TYPE
    ).to('cuda')
    # need to set scheduler to DDIMScheduler for reconstruct from DDIM
    normal_scheduler = DDIMScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')

    # load image 
    image = Image.open('/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val/images/14n_copyroom1/dir_0_mip2.jpg').convert('RGB')

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).to(MASTER_TYPE).unsqueeze(0).permute(0, 3, 1, 2)  # [B, C, H, W]
    image = image * 2 - 1  # normalize to [-1, 1]
    image = image.to('cuda')

    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    prompt_embeds, _ = pipe.encode_prompt(
        prompt = "a printer is sitting on a desk",
        device = image.device,
        num_images_per_prompt=1, 
        do_classifier_free_guidance=False
    )

    noisy_latents = get_ddim_latents(pipe, latents, inverse_scheduler, normal_scheduler, prompt_embeds)
    
    pt_image = generate_images(pipe, noisy_latents=noisy_latents, prompt_embeds=prompt_embeds)
    
    # rescale to [0, 255]
    pt_image = pt_image[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    pt_image = pt_image * 255.0
    pt_image = Image.fromarray(np.clip(pt_image, 0, 255).astype(np.uint8))
    pt_image.save('ddim_reconstructed_image.png')
  




if __name__ == "__main__":
    main()