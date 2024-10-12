import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
import torchvision
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# Useful function for later
def load_image(url, size=None):
    response = requests.get(url, timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img

# Sample function (regular DDIM)

MASTER_TYPE = torch.float16
for guidance_scale in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,1.1,1.2,1.3,1.4,1.6,1.7,1.8,1.9,2.1,2.2,2.3,2.4,2.6,2.7,2.8,2.9,3.1,3.2,3.3,3.4,3.6,3.7,3.8,3.9,4.1,4.2,4.3,4.4,4.6,4.7,4.8,4.9,5.1,5.2,5.3,5.4,5.6,5.7,5.8,5.9,6.1,6.2,6.3,6.4,6.6,6.7,6.8,6.9,7.1,7.2,7.3,7.4,7.6,7.7,7.8,7.9]:
    with torch.inference_mode():
        TOTAL_STEP = 999
        GUIDANCE_SCALE = guidance_scale
        ext = "_torch2.4.1_hf"

        # load stable diffusion
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=MASTER_TYPE).to(device)

        # prepare scheduler in both ways
        normal_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

        input_image = Image.open("input_dog.jpeg").resize((512, 512))
        input_image_prompt = "Photograph of a puppy on the grass"

        # Encode with VAE
        with torch.no_grad():
            latent = pipe.vae.encode((tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)* 2 - 1).half())
            latent = 0.18215 * latent.latent_dist.sample()

        # apply inverse scheduler
        pipe.scheduler = inverse_scheduler

        # DDIM Inversion
        z0_noise = latent
        ddim_latents = []
        inverted_timesteps = []

        def callback_ddim(pipe, step_index, timestep, callback_kwargs):
            ddim_latents.append(callback_kwargs['latents'])
            inverted_timesteps.append(timestep[None])
            return callback_kwargs
        
        generator = torch.Generator(device=device).manual_seed(42)
        ddim_args = {
            "prompt": input_image_prompt,
            "guidance_scale": GUIDANCE_SCALE,
            "latents": z0_noise,
            "output_type": 'latent',
            "return_dict": False,
            "num_inference_steps": TOTAL_STEP,
            "callback_on_step_end": callback_ddim,
            "generator": generator,
        }

        zt_noise, _ = pipe(**ddim_args)
        pipe.scheduler = normal_scheduler

        
        # DDIM generation
        sd_latents = []
        denoised_timesteps = []
        
        def callback_sd(pipe, step_index, timestep, callback_kwargs):
            sd_latents.append(callback_kwargs['latents'])
            denoised_timesteps.append(timestep[None])
            return callback_kwargs
        
        generator = torch.Generator(device=device).manual_seed(42)
        sd_args = {
            "prompt": input_image_prompt,
            "guidance_scale": GUIDANCE_SCALE,
            "latents": zt_noise,
            "output_type": 'latent',
            "return_dict": False,
            "num_inference_steps": TOTAL_STEP,
            "callback_on_step_end": callback_sd,
            "generator": generator,
        }

        z0_output, _ = pipe(**sd_args)


        # for plot  the graph (X-axis)
        ddim_latents = ddim_latents[::-1]
        #inverted_timesteps = inverted_timesteps[::-1]
        inverted_timesteps = denoised_timesteps

        denoised_timesteps = torch.cat(denoised_timesteps)
        inverted_timesteps = torch.cat(inverted_timesteps)

        

        # compute latent meen to plot graph (Y-axis)
        inverted_latents = torch.cat(ddim_latents)
        inverted_latents_mean = inverted_latents.reshape(inverted_latents.size(0),-1).mean(dim=1)

        denoised_latents = torch.cat(sd_latents)
        denoised_latents_mean = denoised_latents.reshape(denoised_latents.size(0),-1).mean(dim=1)

        #decode latent back to image
        output_image = pipe.vae.decode(z0_output / 0.18215).sample.detach().cpu()
        output_image = (output_image + 1) / 2
        output_image = torch.clamp(output_image, 0, 1)

        #convert to PIL image
        output_image = torchvision.transforms.functional.to_pil_image(output_image[0].cpu())

        
        output_image.save(f"output/rgb_g{GUIDANCE_SCALE}_step{TOTAL_STEP}{ext}.png")
        # plot mean 
        plt.plot(inverted_timesteps.cpu().numpy(), inverted_latents_mean.cpu().numpy(), label="Inverted Latents")
        plt.plot(denoised_timesteps.cpu().numpy(), denoised_latents_mean.cpu().numpy(), label="Denoised Latents")
        plt.legend()
        plt.xlabel("Timesteps")
        plt.ylabel("Latents Mean")
        plt.title(f"Latents Mean: Guidance Scale {GUIDANCE_SCALE}, Inferencing Steps {TOTAL_STEP}")
        plt.savefig(f"output/plot_g{GUIDANCE_SCALE}_step{TOTAL_STEP}{ext}.png")
