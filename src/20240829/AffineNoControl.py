"""
AffineDepth.py
Affine transform (Adaptive group norm) that condition with environment map passthrough the VAE 
This version also compute the with depth condition to help the network guide where the light soruce 
"""

import torch 
from AffineControl import AffineControl
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
 
MASTER_TYPE = torch.float16
 
class AffineNoControl(AffineControl):
   
    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5"):
        # load controlnet from pretrain
        

        # load pipeline
        self.pipe =  StableDiffusionPipeline.from_pretrained(
            sd_path,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )
        
        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)