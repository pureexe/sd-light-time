"""
AffineNormal.py
Affine transform (Adaptive group norm) that condition with environment map passthrough the VAE 
This version also compute the with normal map condition to help the network guide where the light soruce 
"""

import torch 
from AffineControl import AffineControl
 
MASTER_TYPE = torch.float16
 
class AffineNormal(AffineControl):
   
    def setup_sd(self):
        super().setup_sd(sd_path="runwayml/stable-diffusion-v1-5", controlnet_path="lllyasviel/sd-controlnet-depth")

    def get_control_image(self, batch):
        assert torch.all(batch['control_normal'] >= 0.0) and torch.all(batch['control_normal'] <= 1.0)
        return batch['control_normal']