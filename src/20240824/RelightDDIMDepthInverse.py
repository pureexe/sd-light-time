# checkpoint is compatible with EnvmapAffine (20240703)

import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
from tqdm.auto import tqdm
from PIL import Image

from transformers import pipeline as transformer_pipeline

from constants import *
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from RelightDDIMInverse import RelightDDIMInverse
from LightEmbedingBlock import set_light_direction
from DDIMInversionForControlNet import DDIMInversionForControlNet
 
class RelightDDIMDepthInverse(RelightDDIMInverse):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__()

        # load controlnet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None, torch_dtype=torch.float16
        )
        # move model to device
        pipe = pipe.to('cuda')
        # reload unet and scheduler
        pipe.unet = self.pipe.unet
        pipe.scheduler = self.pipe.scheduler
        # assign to self
        self.pipe = pipe

        # disable gradient
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        # create ddim inversion
        self.ddim_inversion = DDIMInversionForControlNet(self.pipe)
        self.use_ddim_inversion = True

    def disable_ddim_inversion(self):
        self.use_ddim_inversion = False

    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        GUIDANCE_SCALE = 3 # we temporary use 1, will make this thing scalable in future
        # Apply the source light direction
        # assert image shouldn't be entirely black
        assert torch.any(torch.abs(batch['control_depth'] - (-1)) > 1e-3)

        with torch.inference_mode():
            text_embbeding = self.get_text_embeddings(batch['text'])

        if self.use_ddim_inversion:
            set_light_direction(
                self.pipe.unet, 
                self.get_light_features(
                    batch['source_ldr_envmap'],
                    batch['source_norm_envmap'],
                ), 
                is_apply_cfg=GUIDANCE_SCALE > 1
            )
            # we first find the z0_noise before doing inversion
            with torch.inference_mode():
                z0_noise = self.pipe.vae.encode(batch['source_image']).latent_dist.sample() * self.pipe.vae.config.scaling_factor

            # DDIM inverse to get an intial noise
            zt_noise, _ = self.ddim_inversion(
                z0_noise, 
                text_embbeding, 
                self.target_timestep, 
                self.num_inference_steps,
                device=z0_noise.device,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_cond=batch['control_depth'],
                cond_scale=1
            )
        else:
            zt_noise = None

        # Apply the target light direction
        set_light_direction(
            self.pipe.unet,
            self.get_light_features(
                batch['target_ldr_envmap'],
                batch['target_norm_envmap']
            ),
            is_apply_cfg=GUIDANCE_SCALE > 1
        )

        pt_image, _ = self.pipe(
            image = batch['control_depth'],
            prompt_embeds=text_embbeding, 
            latents=zt_noise,
            negative_prompt="",
            output_type="pt",
            guidance_scale=GUIDANCE_SCALE, 
            num_inference_steps=self.num_inference_steps,
            return_dict = False,
            generator=torch.Generator().manual_seed(42)
        )

        gt_image = (batch["source_image"] + 1.0) / 2.0
        images = torch.cat([gt_image, pt_image], dim=0)
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
        if is_save_image:
            os.makedirs(f"{self.logger.log_dir}/with_groudtruth", exist_ok=True)
            torchvision.utils.save_image(image, f"{self.logger.log_dir}/with_groudtruth/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
            os.makedirs(f"{self.logger.log_dir}/crop_image", exist_ok=True)
            torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/crop_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)

