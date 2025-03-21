"""
AffineControl.py
Affine transform (Adaptive group norm) 
"""

import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from tqdm.auto import tqdm
import ezexr
from PIL import Image

from constants import *
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler, DDIMInverseScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from DDIMInversion import DDIMInversion
from LightEmbedingBlock import set_light_direction, add_light_block, set_gate_shift_scale

from ball_helper import inpaint_chromeball, pipeline2controlnetinpaint
 
MASTER_TYPE = torch.float16
 
class AffineControl(L.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            guidance_scale=3.0,
            gate_multipiler=1,
            feature_type="vae",
            num_inversion_steps=200,
            num_inference_steps=50,
            *args,
            **kwargs
        ) -> None:
        super().__init__()
        self.gate_multipiler = gate_multipiler
        self.condition_scale = 1.0
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.feature_type = feature_type
        self.num_inversion_steps = num_inversion_steps
        self.num_inference_steps = num_inference_steps
        self.save_hyperparameters()
        

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
        self.setup_ddim()
        self.setup_light_block()


    def setup_light_block(self):
        if self.feature_type == "shcoeff_order2":
            mlp_in_channel = 27
        elif self.feature_type == "vae":
            mlp_in_channel = 32*32*4*2
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        #  add light block to unet, 1024 is the shape of output of both LDR and HDR_Normalized clip combine
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')

        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad and (len(p.shape) > 1 or p.shape[0] > 1), self.pipe.unet.parameters())
        gate_trainable = filter(lambda p: p.requires_grad and (len(p.shape) == 1 and p.shape[0] == 1), self.pipe.unet.parameters())

        self.unet_trainable = torch.nn.ParameterList(unet_trainable)
        self.gate_trainable = torch.nn.ParameterList(gate_trainable)

    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path="lllyasviel/sd-controlnet-depth"):
        # load controlnet from pretrain
        
        # check if controlnet_path is list
        if isinstance(controlnet_path, list):
            controlnet = MultiControlNetModel([
                ControlNetModel.from_pretrained(path, torch_dtype=MASTER_TYPE) for path in controlnet_path
            ])
            self.condition_scale = [1.0] * len(controlnet_path)
        else:
            controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=MASTER_TYPE)

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            controlnet=controlnet,
            safety_checker=None, torch_dtype=MASTER_TYPE
        )

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.controlnet.requires_grad_(False)
        #self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

        # load pipe_chromeball for validation 
        #controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=MASTER_TYPE)
        #self.pipe_chromeball = pipeline2controlnetinpaint(self.pipe, controlnet=controlnet_depth).to('cuda')

    def setup_ddim(self):
        self.normal_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        self.pipe.scheduler = self.normal_scheduler
        self.ddim_inversion = DDIMInversion(self.pipe)
        
    
    def set_seed(self, seed):
        self.seed = seed

    def set_gate_shift_scale(self, gate_shift, gate_scale):
        set_gate_shift_scale(self.pipe.unet, gate_shift, gate_scale)
   
    def get_vae_features(self, images, generator=None):
        assert images.shape[1] == 3, "Only support RGB image"
        assert images.shape[2] == 256 and images.shape[3] == 256, "Only support 256x256 image"
        with torch.inference_mode():
            # VAE need input in range of [-1,1]
            images = images * 2.0 - 1.0
            emb = self.pipe.vae.encode(images).latent_dist.sample(generator=generator) * self.pipe.vae.config.scaling_factor 
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb
        
    def get_light_features(self, batch, array_index=None, generator=None):
        if self.feature_type == "vae":
            ldr_envmap = batch['ldr_envmap']
            norm_envmap = batch['norm_envmap']
            if array_index is not None:
                ldr_envmap = ldr_envmap[array_index]
                norm_envmap = norm_envmap[array_index]
            ldr_features = self.get_vae_features(ldr_envmap, generator)
            hdr_features = self.get_vae_features(norm_envmap, generator)
            # concat ldr and hdr features
            return torch.cat([ldr_features, hdr_features], dim=-1)    
        elif self.feature_type == "shcoeff_order2":
            shcoeff = batch['sh_coeffs']
            if array_index is not None:
                shcoeff = shcoeff[array_index]
            shcoeff = shcoeff.view(shcoeff.size(0), -1) #flatten shcoeff to [B, 27]
            return shcoeff
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
    
    def compute_train_loss(self, batch, batch_idx, timesteps=None, seed=None):
        text_inputs = self.pipe.tokenizer(
                batch['text'],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids


        latents = self.pipe.vae.encode(batch['source_image']).latent_dist.sample().detach()

        latents = latents * self.pipe.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        if seed is not None:
            # create random genration seed
            torch.manual_seed(seed)
            noise = torch.randn_like(latents, memory_format=torch.contiguous_format)
        else:
            noise = torch.randn_like(latents)
        target = noise 
    
        bsz = latents.shape[0]

        if timesteps is None:
            timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long().to(latents.device)
        else:
            if isinstance(timesteps, int):
                timesteps = torch.tensor([timesteps], device=latents.device)
            timesteps = timesteps.expand(bsz)
            timesteps = timesteps.long().to(latents.device)


        text_input_ids = text_input_ids.to(latents.device)
        encoder_hidden_states = self.pipe.text_encoder(text_input_ids)[0]

        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # set light direction        
        light_features = self.get_light_features(batch)
        set_light_direction(self.pipe.unet, light_features, is_apply_cfg=False) #B,C

        if hasattr(self.pipe,"controlnet"): #use controlnet when computing loss
            # forward the controlnet
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=self.get_control_image(batch),
                conditioning_scale=self.condition_scale,
                guess_mode=False,
                return_dict=False,
            )
            
            model_pred = self.pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
        else:
            model_pred = self.pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]


        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def get_control_image(self, batch):
        raise NotImplementedError("get_control_image must be implemented")
    
    def select_batch_keyword(self, batch, keyword):
                    
        if self.feature_type == "vae":
            batch['ldr_envmap'] = batch[f'{keyword}_ldr_envmap']
            batch['norm_envmap'] = batch[f'{keyword}_norm_envmap']
        elif self.feature_type == "shcoeff_order2":
            batch['sh_coeffs'] = batch[f'{keyword}_sh_coeffs']
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        return batch
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):
        USE_LIGHT_DIRECTION_CONDITION = True
        USE_OWN_DDIM = True

        is_apply_cfg = self.guidance_scale > 1
        
        # Apply the source light direction
        self.select_batch_keyword(batch, 'source')
        if USE_LIGHT_DIRECTION_CONDITION:
            set_light_direction(
                self.pipe.unet, 
                self.get_light_features(batch), 
                is_apply_cfg=is_apply_cfg
            )
        else:
            set_light_direction(
            self.pipe.unet, 
                None, 
                is_apply_cfg=False
            )

        # we first find the z0_noise before doing inversion
        negative_embedding = None
        with torch.inference_mode():
            z0_noise = self.pipe.vae.encode(batch['source_image']).latent_dist.sample(generator=torch.Generator().manual_seed(self.seed)) * self.pipe.vae.config.scaling_factor
            text_embbeding = self.get_text_embeddings(batch['text'])
            if self.guidance_scale > 1:
                negative_embedding = self.get_text_embeddings([''])
                negative_embedding = negative_embedding.repeat(text_embbeding.shape[0], 1, 1)

        if USE_OWN_DDIM:
            ddim_args = {
                'z': z0_noise,
                'embedd': text_embbeding,
                'negative_embedd': negative_embedding,   
                'target_timestep': 1000,
                'num_inference_steps': self.num_inversion_steps,
                'guidance_scale': 1.0,
                'device': z0_noise.device,
            }
            if hasattr(self.pipe, 'controlnet'):
                ddim_args['controlnet_cond'] = self.get_control_image(batch)    
                ddim_args['cond_scale'] = self.condition_scale
                # DDIM inverse to get an intial noise
            zt_noise, _ = self.ddim_inversion(**ddim_args)
        else:
            self.pipe.scheduler = self.inverse_scheduler
            zt_noise, _ = self.pipe(
                prompt_embeds=text_embbeding,
                negative_prompt_embeds=negative_embedding, 
                guidance_scale=self.guidance_scale,
                latents=z0_noise,
                output_type='latent',
                return_dict=False,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator().manual_seed(self.seed)
            )
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch)
            self.pipe.scheduler = self.nomral_scheduler

        # if dataset is not list, convert to list
        for key in ['target_ldr_envmap', 'target_norm_envmap', 'target_image', 'target_sh_coeffs', 'word_name']:
            if key in batch and not isinstance(batch[key], list):
                batch[key] = [batch[key]]


        if USE_LIGHT_DIRECTION_CONDITION:
            #Apply the target light direction
            self.select_batch_keyword(batch, 'target')    
        
       

        # compute inpaint from nozie
        mse_output = []
        for target_idx in range(len(batch['target_image'])):
            set_light_direction(
                self.pipe.unet,
                self.get_light_features(batch, array_index=target_idx),
                is_apply_cfg=is_apply_cfg
            )
            pipe_args = {
                "prompt_embeds": text_embbeding,
                "negative_prompt_embeds": negative_embedding,
                "latents": zt_noise.clone(),
                "output_type": "pt",
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed)
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch)
            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            pt_image, _ = self.pipe(**pipe_args)
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0
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
            
            
            tb_name = f"{batch['name'][0].replace('/','-')}/{batch['word_name'][target_idx][0].replace('/','-')}"
            
            self.logger.experiment.add_image(f'{tb_name}', image, self.global_step)

            # calcuarte psnr 
            mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
            mse_output.append(mse[None])
            psnr = -10 * torch.log10(mse)
            self.log('psnr', psnr)
            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                epoch_text = f"epoch_{self.current_epoch:04d}/" if is_seperate_dir_with_epoch else ""

                # save with ground truth
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}with_groudtruth", exist_ok=True)
                torchvision.utils.save_image(image, f"{self.logger.log_dir}/{epoch_text}with_groudtruth/{filename}.jpg")

                # save image file
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/{epoch_text}crop_image/{filename}.jpg")
                # save psnr to file
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}psnr", exist_ok=True)
                with open(f"{self.logger.log_dir}/{epoch_text}psnr/{filename}.txt", "w") as f:
                    f.write(f"{psnr.item()}\n")
                # save controlnet 
                if hasattr(self.pipe, "controlnet"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}control_image", exist_ok=True)
                    if isinstance(ctrl_image, list):
                        for i, c in enumerate(ctrl_image):
                            torchvision.utils.save_image(c, f"{self.logger.log_dir}/{epoch_text}control_image/{filename}_{i}.jpg")
                    else:
                        torchvision.utils.save_image(ctrl_image, f"{self.logger.log_dir}/{epoch_text}control_image/{filename}.jpg")
                # save the target_ldr_envmap
                if hasattr(batch, "target_ldr_envmap"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}target_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_ldr_envmap'][target_idx], f"{self.logger.log_dir}/{epoch_text}target_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "target_norm_envmap"):
                    # save the target_norm_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}target_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_norm_envmap'][target_idx], f"{self.logger.log_dir}/{epoch_text}target_norm_envmap/{filename}.jpg")
                if hasattr(batch, "source_ldr_envmap"):
                    # save the source_ldr_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_ldr_envmap'], f"{self.logger.log_dir}/{epoch_text}source_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "source_norm_envmap"):
                    # save the source_norm_envmap
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_norm_envmap'], f"{self.logger.log_dir}/{epoch_text}source_norm_envmap/{filename}.jpg")
                if hasattr(self, "pipe_chromeball"):
                    os.makedirs(f"{self.logger.log_dir}/{epoch_text}inpainted_image", exist_ok=True)
                    torchvision.utils.save_image(inpainted_image, f"{self.logger.log_dir}/{epoch_text}inpainted_image/{filename}.jpg")
                # save prompt
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}prompt", exist_ok=True) 
                with open(f"{self.logger.log_dir}/{epoch_text}prompt/{filename}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{self.logger.log_dir}/{epoch_text}source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{self.logger.log_dir}/{epoch_text}source_image/{filename}.jpg")
            if batch_idx == 0:
                if hasattr(self, "gate_trainable"):
                    # plot gate_trainable
                    for gate_id, gate in enumerate(self.gate_trainable):
                        self.logger.experiment.add_scalar(f'gate/{gate_id:02d}', gate, self.global_step)
                        self.logger.experiment.add_scalar(f'gate/average', gate, self.global_step)
            if self.global_step == 0:
                self.logger.experiment.add_text(f'text/{batch["word_name"][0]}', batch['text'][0], self.global_step)
            if self.global_step == 0 and batch_idx == 0:
                self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
                if hasattr(self, "gate_multipiler"):
                    self.logger.experiment.add_text('gate_multipiler', str(self.gate_multipiler), self.global_step)
        mse_output = torch.cat(mse_output, dim=0)
        mse_output = mse_output.mean()
        return mse_output
                
    def test_step(self, batch, batch_idx):
        if self.is_plot_train_loss:
            self.plot_train_loss(batch, batch_idx, is_save_image=True, seed=self.seed)
        else:
            self.generate_tensorboard(batch, batch_idx, is_save_image=True)
    
    def disable_plot_train_loss(self):
        self.is_plot_train_loss = False
    
    def enable_plot_train_loss(self):
        self.is_plot_train_loss = True

    #TODO: let's create seperate file that take checkpoint and compute the loss. this loss code should be re-implememet
    def plot_train_loss(self, batch, batch_idx, is_save_image=False, seed=None):
        for timestep in range(100, 1000, 100):
            loss = self.compute_train_loss(batch, batch_idx, timesteps=timestep, seed=seed)
            # TODO: let's check the log and make sure it do average.
            self.logger.experiment.add_scalar(f'plot_train_loss/{timestep}', loss, self.global_step)
            self.logger.experiment.add_scalar(f'plot_train_loss/average', loss, self.global_step)
            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][0].replace('/','-')}"
                os.makedirs(f"{self.logger.log_dir}/train_loss/{timestep}", exist_ok=True)
                with open(f"{self.logger.log_dir}/train_loss/{timestep}/{filename}.txt", "w") as f:
                    f.write(f"{loss.item()}")

    def validation_step(self, batch, batch_idx):
        self.select_batch_keyword(batch, 'source')
        self.plot_train_loss(batch, batch_idx, seed=None) # DEFAULT USE SEED 42
        mse = self.generate_tensorboard(batch, batch_idx, is_save_image=True, is_seperate_dir_with_epoch=True) #NOTE: temporary enable save image for predict the light
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.unet_trainable, 'lr': self.learning_rate},
            {'params': self.gate_trainable, 'lr': self.learning_rate * self.gate_multipiler}
        ])
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale

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
