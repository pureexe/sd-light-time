"""
AffineDepth.py
Affine transform (Adaptive group norm) that condition with environment map passthrough the VAE 
This version also compute the with depth condition to help the network guide where the light soruce 
"""

import torch 
from AffineControl import AffineControl
 
MASTER_TYPE = torch.float16
 
class AffineDepthNormalBae(AffineControl):
   

    def setup_sd(self):
        controlnet_depth_path = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet_normal_path = "lllyasviel/control_v11p_sd15_normalbae"
        super().setup_sd(
            sd_path="runwayml/stable-diffusion-v1-5",
            controlnet_path=[controlnet_depth_path, controlnet_normal_path]
        )


    def get_control_image(self, batch):
        assert torch.all(batch['control_depth'] >= 0) and torch.all(batch['control_depth'] <= 1)
        assert torch.all(batch['control_normal_bae'] >= 0) and torch.all(batch['control_normal_bae'] <= 1)
        return [batch['control_depth'], batch['control_normal_bae']]