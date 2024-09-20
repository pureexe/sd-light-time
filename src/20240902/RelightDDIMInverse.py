
import os 
import torch 
import torchvision

from constants import *
from diffusers import DDIMScheduler

from LightEmbedingBlock import set_light_direction
from DDIMInversion import DDIMInversion
from AffineControl import AffineControl
from ball_helper import inpaint_chromeball
import numpy as np

 

def create_ddim_inversion(base_class):
    
    class RelightDDIMInverse(base_class):
        def __init__(self, *args,  **kwargs) -> None:
            super().__init__()

            # DEFAULT VARIABLE 
            self.target_timestep = 1000
            self.num_inference_steps = 200 # may need to use many steps to get the best result

            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')

            # create ddim inversion
            self.ddim_inversion = DDIMInversion(self.pipe)
            
            # dsiable chromeball
            #del self.pipe_chromeball

        def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
            # Apply the source light direction
            set_light_direction(
                self.pipe.unet, 
                self.get_light_features(
                    batch['source_ldr_envmap'],
                    batch['source_norm_envmap'],
                ), 
                is_apply_cfg=self.guidance_scale > 1
            )
            
            # we first find the z0_noise before doing inversion
            with torch.inference_mode():
                z0_noise = self.pipe.vae.encode(batch['source_image']).latent_dist.sample() * self.pipe.vae.config.scaling_factor
                text_embbeding = self.get_text_embeddings(batch['text'])

            ddim_args = {
                'z': z0_noise,
                'embedd': text_embbeding,
                'target_timestep': self.target_timestep,
                'num_inference_steps': self.num_inference_steps,
                'guidance_scale': self.guidance_scale,
                'device': z0_noise.device,
            }
            if hasattr(self.pipe, 'controlnet'):
                ddim_args['controlnet_cond'] = self.get_control_image(batch)    
                ddim_args['cond_scale'] = self.condition_scale

            # DDIM inverse to get an intial noise
            zt_noise, _ = self.ddim_inversion(**ddim_args)

            # Apply the target light direction
            set_light_direction(
                self.pipe.unet,
                self.get_light_features(
                    batch['target_ldr_envmap'],
                    batch['target_norm_envmap']
                ),
                is_apply_cfg=self.guidance_scale > 1
            )



            pipe_args = {
                "prompt_embeds": text_embbeding,
                "negative_prompt": "",
                "latents": zt_noise,
                "output_type": "pt",
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed)
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch)
            pt_image, _ = self.pipe(**pipe_args)
            gt_image = (batch["source_image"] + 1.0) / 2.0
            tb_image = [gt_image, pt_image]

            if hasattr(self.pipe, "controlnet"):
                ctrl_image = self.get_control_image(batch)
                if isinstance(ctrl_image, list):
                    tb_image += ctrl_image
                else:
                    tb_image.append(ctrl_image)

            if hasattr(self, "pipe_chromeball"):
                with torch.inference_mode():
                    # convert pt_image to pil_image
                    to_inpaint_img = torchvision.transforms.functional.to_pil_image(pt_image[0].cpu())                
                    inpainted_image = inpaint_chromeball(to_inpaint_img,self.pipe_chromeball)
                    inpainted_image = torchvision.transforms.functional.to_tensor(inpainted_image).to(pt_image.device)
                    tb_image.append(inpainted_image[None])

           
            images = torch.cat(tb_image, dim=0)
            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
            
            self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
            if is_save_image:
                os.makedirs(f"{self.logger.log_dir}/with_groudtruth", exist_ok=True)
                torchvision.utils.save_image(image, f"{self.logger.log_dir}/with_groudtruth/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                if hasattr(self.pipe, "controlnet"):
                    os.makedirs(f"{self.logger.log_dir}/control_image", exist_ok=True)
                    if isinstance(ctrl_image, list):
                        for i, c in enumerate(ctrl_image):
                            torchvision.utils.save_image(c, f"{self.logger.log_dir}/control_image/{batch['name'][0]}_{batch['word_name'][0]}_{i}.jpg")
                    else:
                        torchvision.utils.save_image(ctrl_image, f"{self.logger.log_dir}/control_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                os.makedirs(f"{self.logger.log_dir}/crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/crop_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the target_ldr_envmap
                os.makedirs(f"{self.logger.log_dir}/target_ldr_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['target_ldr_envmap'], f"{self.logger.log_dir}/target_ldr_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the target_norm_envmap
                os.makedirs(f"{self.logger.log_dir}/target_norm_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['target_norm_envmap'], f"{self.logger.log_dir}/target_norm_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the source_ldr_envmap
                os.makedirs(f"{self.logger.log_dir}/source_ldr_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['source_ldr_envmap'], f"{self.logger.log_dir}/source_ldr_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the source_norm_envmap
                os.makedirs(f"{self.logger.log_dir}/source_norm_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['source_norm_envmap'], f"{self.logger.log_dir}/source_norm_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                if hasattr(self, "pipe_chromeball"):
                    os.makedirs(f"{self.logger.log_dir}/inpainted_image", exist_ok=True)
                    torchvision.utils.save_image(inpainted_image, f"{self.logger.log_dir}/inpainted_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save prompt
                os.makedirs(f"{self.logger.log_dir}/prompt", exist_ok=True) 
                with open(f"{self.logger.log_dir}/prompt/{batch['name'][0]}_{batch['word_name'][0]}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{self.logger.log_dir}/source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{self.logger.log_dir}/source_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
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

    return RelightDDIMInverse