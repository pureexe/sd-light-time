"""
AffineDepth.py
Affine transform (Adaptive group norm) that condition with environment map passthrough the VAE 
This version also compute the with depth condition to help the network guide where the light soruce 
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
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from LightEmbedingBlock import set_light_direction, add_light_block, set_gate_shift_scale
from UnsplashLiteDataset import log_map_to_range

from ball_helper import inpaint_chromeball, pipeline2controlnetinpaint
 
MASTER_TYPE = torch.float16
 
class AffineControl(L.LightningModule):

    def __init__(self, learning_rate=1e-4, guidance_scale=3.0, gate_multipiler=1, feature_type="shcoeff_order2", *args, **kwargs) -> None:
        super().__init__()
        self.gate_multipiler = gate_multipiler
        self.condition_scale = 1.0
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.feature_type = feature_type

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
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
        controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=MASTER_TYPE)
        self.pipe_chromeball = pipeline2controlnetinpaint(self.pipe, controlnet=controlnet_depth).to('cuda')
        
    
    def set_seed(self, seed):
        self.seed = seed

    def set_gate_shift_scale(self, gate_shift, gate_scale):
        set_gate_shift_scale(self.pipe.unet, gate_shift, gate_scale)
   
    def get_vae_features(self, images, generator=None):
        assert images.shape[1] == 3, "Only support RGB image"
        assert images.shape[2] == 256 and images.shape[3] == 256, "Only support 256x256 image"
        #assert images.min() >= 0.0 and images.max() <= 1.0, "Only support [0, 1] range image"
        with torch.inference_mode():
            # VAE need input in range of [-1,1]
            images = images * 2.0 - 1.0
            emb = self.pipe.vae.encode(images).latent_dist.sample(generator=generator) * self.pipe.vae.config.scaling_factor 
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb
        
    def get_light_features(self, batch, generator=None):
        if self.feature_type == "vae":
            ldr_features = self.get_vae_features(batch['ldr_envmap'], generator)
            hdr_features = self.get_vae_features(batch['norm_envmap'], generator)
            # concat ldr and hdr features
            return torch.cat([ldr_features, hdr_features], dim=-1)    
        elif self.feature_type == "shcoeff_order2":
            shcoeff = batch['sh_coeffs']
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
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        set_light_direction(self.pipe.unet, self.get_light_features(batch), is_apply_cfg=False)
        
        # control_depth should not be entirely black (all -1)
        pipe_args = {
            "prompt": batch['text'],
            "output_type": "pt",
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": 50,
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
        
        self.logger.experiment.add_image(f'{batch["name"][0]}', image, self.global_step)
        # calcuarte psnr 
        mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
        psnr = -10 * torch.log10(mse)
        self.log('psnr', psnr)
        if is_save_image:
            os.makedirs(f"{self.logger.log_dir}/crop_image", exist_ok=True)
            torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/crop_image/{batch['word_name'][0]}.jpg")
            # save psnr to file
            os.makedirs(f"{self.logger.log_dir}/psnr", exist_ok=True)
            with open(f"{self.logger.log_dir}/psnr/{batch['word_name'][0]}.txt", "w") as f:
                f.write(f"{psnr.item()}\n")
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
        return mse
                
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
                os.makedirs(f"{self.logger.log_dir}/train_loss/{timestep}", exist_ok=True)
                with open(f"{self.logger.log_dir}/train_loss/{timestep}/{batch['name'][0]}.txt", "w") as f:
                    f.write(f"{loss.item()}")

    def validation_step(self, batch, batch_idx):
        self.plot_train_loss(batch, batch_idx, seed=None) # DEFAULT USE SEED 42
        mse = self.generate_tensorboard(batch, batch_idx, is_save_image=False)
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
