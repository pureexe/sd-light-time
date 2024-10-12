# checkpoint is compatible with EnvmapAffine (20240703)

import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from tqdm.auto import tqdm
import ezexr

from constants import *
from diffusers import StableDiffusionPipeline, DDIMScheduler

from LightEmbedingBlock import set_light_direction, add_light_block
from DDIMInversion import DDIMInversion
from AffineConsistancy26 import AffineConsistancy26


 
 
class RelightDDIMInverse26(AffineConsistancy26):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__()

        # DEFAULT VARIABLE 
        self.target_timestep = 1000
        self.num_inference_steps = 200 # may need to use many steps to get the best result

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')

        # create ddim inversion
        self.ddim_inversion = DDIMInversion(self.pipe)


        
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        GUIDANCE_SCALE = 1 # we temporary use 1, will make this thing scalable in future
        # Apply the source light direction
        set_light_direction(
            self.pipe.unet, 
            self.get_light_features(
                batch['source_ldr_envmap'],
                batch['source_norm_envmap'],
                # batch['target_ldr_envmap'],
                # batch['target_norm_envmap']
            ), 
            is_apply_cfg=GUIDANCE_SCALE > 1
        )
        
        # we first find the z0_noise before doing inversion
        with torch.inference_mode():
            z0_noise = self.pipe.vae.encode(batch['source_image']).latent_dist.sample() * self.pipe.vae.config.scaling_factor
            print(batch['text'])
            text_embbeding = self.get_text_embeddings(batch['text'])


        # DDIM inverse to get an intial noise
        zt_noise, _ = self.ddim_inversion(
            z0_noise, 
            text_embbeding, 
            self.target_timestep, 
            self.num_inference_steps,
            device=z0_noise.device,
            guidance_scale=GUIDANCE_SCALE
        )

        # Apply the target light direction
        set_light_direction(
            self.pipe.unet,
            self.get_light_features(
                # batch['source_ldr_envmap'],
                # batch['source_norm_envmap'],
                batch['target_ldr_envmap'],
                batch['target_norm_envmap']
            ),
            is_apply_cfg=GUIDANCE_SCALE > 1
        )

        pt_image, _ = self.pipe(
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
