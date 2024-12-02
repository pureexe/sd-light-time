"""
sd3adagncontrol.py
SD3 Adaptive group norm
"""
import copy
import os 
import torch 
import torch.utils
import torchvision
import lightning as L
import numpy as np
import torchvision

from constants import *
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, ControlNetModel
from InversionHelper import get_ddim_latents

from TrainableAttentionBlock import Light2TokenBlock, enable_grad_on_kv

MASTER_TYPE = torch.float32

class SDRelightWithChromeball(L.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            guidance_scale=1.0,
            feature_type="shcoeff_order2",
            num_inversion_steps=500,
            num_inference_steps=500,
            num_light_token=1,
            *args,
            **kwargs
        ) -> None:
        super().__init__()
        self.condition_scale = 1.0
        self.guidance_scale = guidance_scale
        self.ddim_guidance_scale = 1.0       
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.feature_type = feature_type
        self.num_light_token = num_light_token

        self.num_inversion_steps = num_inversion_steps
        self.num_inference_steps = num_inference_steps
        self.save_hyperparameters()

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
        self.setup_light_block()
        self.setup_trainable()
        self.log_dir = ""

    def setup_trainable(self):
        # register trainable module to pytorch lighting model 

        # light2token is already auto register

        # register key value
        trainable_kv = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.trainable_kv = torch.nn.ParameterList(trainable_kv)



    def setup_light_block(self):
        if self.feature_type == "shcoeff_order2":
            mlp_in_channel = 27
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        
        enable_grad_on_kv(self.pipe.unet)
        self.pipe.to('cuda')
        self.light2token = Light2TokenBlock(out_dim=768, in_dim=mlp_in_channel, num_token=self.num_light_token)

    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # check if controlnet_path is list
        if controlnet_path is not None:
            raise Exception('controlnet is not support at the moment')
            # if later implement controlnet support, please disable require gradon controlnet also.
        
        # load pipeline
        self.pipe =  StableDiffusionPipeline.from_pretrained(
            sd_path,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=MASTER_TYPE)
        self.depth_controlnet.requires_grad_(False)

        #self.regular_scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

    def set_seed(self, seed):
        self.seed = seed           
        
    def get_light_features(self, batch, array_index=None, generator=None):
        if self.feature_type in ["shcoeff_order2","shcoeff_fuse"]:
            shcoeff = batch['sh_coeffs']
            if array_index is not None:
                shcoeff = shcoeff[array_index]
            shcoeff = shcoeff.view(shcoeff.size(0), -1) #flatten shcoeff to [B, 27]
            return shcoeff
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")

    def get_vae_shiftscale(self):
        shift = self.pipe.vae.config.shift_factor if hasattr(self.pipe.vae.config, 'shift_factor') else 0.0
        scale = self.pipe.vae.config.scaling_factor if hasattr(self.pipe.vae.config, 'scaling_factor') else 1.0
        # clear out None type 
        shift = 0.0 if shift is None else shift
        scale = 1.0 if scale is None else scale
        return shift, scale

    def get_latents_from_images(self, image):
        latents = self.pipe.vae.encode(image).latent_dist.sample().detach()
        shift, scale = self.get_vae_shiftscale()
        latents = (latents - shift) * scale        
        return latents
    
    def get_images_from_latents(self, latents):
        shift, scale = self.get_vae_shiftscale()
        latents = (latents / scale) + shift
        images = self.pipe.vae.decode(latents).sample
        return images
    
    def get_noise_from_latents(self, latents, seed=None):
        """
        Obtain gaussain noise same shape as the latents
        """
        if seed is not None:
            # create random genration seed
            torch.manual_seed(seed)
            noise = torch.randn_like(latents, memory_format=torch.contiguous_format)
        else:
            noise = torch.randn_like(latents)
        return noise             

    def get_timesteps(self, timesteps, batch_size, device):
        """
        Get time step for compute loss 
        """
        if timesteps is None:
            # random timestep for training
            timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device)
            timesteps = timesteps.long().to(device)
        else:
            # get sepecific timestep for compute loss
            if isinstance(timesteps, int):
                timesteps = torch.tensor([timesteps], device=device)
            timesteps = timesteps.expand(batch_size)
            timesteps = timesteps.long().to(device)

        return timesteps
    
    def compute_train_loss(self, batch, batch_idx, timesteps=None, seed=None):
        ori_timesteps = timesteps
        # get device 
        device = batch['source_image'].device
        batch_size = batch['source_image'].shape[0]
        
        # compute text embedding 
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt = batch['text'],
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )

        # compute image original latent
        latents = self.get_latents_from_images(batch['source_image'])
        
        # get gaussain nolise 
        noise = self.get_noise_from_latents(latents)
        target = noise # this noise also be a training target for our model

        # get timesteps for training 
        timesteps = self.get_timesteps(timesteps, batch_size, device)

        # add noise from a clear latents to specific corrupt noise that have match timestep
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

        # set light direction        
        light_features = self.get_light_features(batch).to(latents.dtype)
        light_embeds = self.light2token(light_features)
 
        encoder_hidden_states = torch.cat([prompt_embeds, light_embeds], dim=-2)
        
        down_block_res_samples = None
        mid_block_res_sample = None

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
        
        # compute diffusion loss
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        # let's compute chromeball from the model_pred. first we get next latents by step for 1 time 
        self.pipe.scheduler.set_timesteps(1000) # on training, we do sample 1000 timesteps
        #ball_latents = self.pipe.scheduler.step(model_pred, timesteps, noisy_latents).prev_sample # step t-1
        
        # calcurate pred_x0
        sqrt_alpha_prod = self.pipe.scheduler.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_prod = (1 - self.pipe.scheduler.alphas_cumprod[timesteps]).sqrt()  # sqrt(1 - \bar{\alpha}_t)
        pred_x0 = (latents - sqrt_one_minus_alpha_prod * model_pred) / sqrt_alpha_prod

        # then add timestep back to 900
        BALL_TIMESTEP = 900 
        ball_timesteps = self.get_timesteps(BALL_TIMESTEP, batch_size, device)
        ball_latents = self.pipe.scheduler.add_noise(pred_x0, noise, ball_timesteps)

        ball_guidance_scale = 5.0

        # compute text embedding  each are (1,77,768)
        chromeball_prompt_embeds, chomeball_negative_embeds = self.pipe.encode_prompt(
            prompt = 'a perfect mirrored reflective chrome ball sphere',
            negative_prompt='matte, diffuse, flat, dull',
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True
        )

        if ball_guidance_scale > 1.0:
            chromeball_embeds = torch.cat([chomeball_negative_embeds, chromeball_prompt_embeds], dim=0)
            ball_latents_input = torch.cat([ball_latents, ball_latents])
        else:
            chromeball_embeds = chromeball_prompt_embeds
            ball_latents_input = ball_latents
 
        # compute controlnet
        ball_down_block_res_samples, ball_mid_block_res_sample = self.depth_controlnet(
            ball_latents_input,
            ball_timesteps,
            encoder_hidden_states=chromeball_embeds,
            controlnet_cond=self.get_depth_with_ball(batch),
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
        )

        # compute chromeball
        ball_pred = self.pipe.unet(
            ball_latents_input,
            ball_timesteps,
            encoder_hidden_states=chromeball_embeds,
            down_block_additional_residuals=ball_down_block_res_samples,
            mid_block_additional_residual=ball_mid_block_res_sample,
            return_dict=False,
        )[0]

        if ball_guidance_scale > 1.0:
            ball_pred_uncond, ball_pred_text = ball_pred.chunk(2)
            ball_pred = ball_pred_uncond + ball_guidance_scale * (ball_pred_text - ball_pred_uncond)

        sqrt_alpha_prod = self.pipe.scheduler.alphas_cumprod[ball_timesteps].sqrt()
        sqrt_one_minus_alpha_prod = (1 - self.pipe.scheduler.alphas_cumprod[ball_timesteps]).sqrt()  # sqrt(1 - \bar{\alpha}_t)

        ball_x0 = (ball_latents - sqrt_one_minus_alpha_prod * ball_pred) / sqrt_alpha_prod

        ball_x0_img = self.get_images_from_latents(ball_x0)

        filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][0][0].replace('/','-')}"
        print("Processing: ", batch['name'] , "/", batch['word_name'])
        # save image file
        epoch_text = f"epoch_{self.current_epoch:04d}/"
        logdir = self.logger.log_dir
        os.makedirs(f"{logdir}/{epoch_text}from_prompt", exist_ok=True)
        for x0_id in range(ball_x0_img.shape[0]):
           ball_x0_img[x0_id] = (torch.clamp(ball_x0_img[x0_id], -1, 1) + 1.0) / 2.0
           torchvision.utils.save_image(ball_x0_img[x0_id], f"{logdir}/{epoch_text}from_prompt/{filename}_{ori_timesteps}_{x0_id}.png")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def get_control_image(self, batch, array_index=None):
        raise NotImplementedError("get_control_image must be implmenented")
    
    def get_depth_with_ball(self, batch):
        return set_circle_in_tensor(batch['control_depth'])
                
    def select_batch_keyword(self, batch, keyword):
                    
        if self.feature_type in ["shcoeff_order2", "shcoeff_fuse"]:
            batch['sh_coeffs'] = batch[f'{keyword}_sh_coeffs']
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        return batch
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):    
        return None
        try:
            log_dir = self.logger.log_dir
        except:
            log_dir = self.log_dir

        USE_LIGHT_DIRECTION_CONDITION = True
        USE_FROM_PROMPT_GENERATION = True

        # precompute-variable
        is_apply_cfg = self.guidance_scale > 1
        # _step_{self.global_step:06d}
        epoch_text = f"epoch_{self.current_epoch:04d}/" if is_seperate_dir_with_epoch else ""
        source_name = f"{batch['name'][0].replace('/','-')}"
        device = batch['source_image'].device

        # Apply the source light direction
        self.select_batch_keyword(batch, 'source')

        # set direction for inversion        
        source_light_features = self.get_light_features(batch, generator=torch.Generator().manual_seed(self.seed))
        source_light_features = source_light_features if USE_LIGHT_DIRECTION_CONDITION else None
        source_light_embeds = self.light2token(source_light_features)

        # Inversion to get z_noise 
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt = batch['text'],
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=is_apply_cfg,
            negative_prompt = [""] * len(batch['text'])
        )

        source_embeds = torch.cat([prompt_embeds, source_light_embeds], dim=-2)

        
        # get DDIM inversion  
        ddim_latents, ddim_timesteps = get_ddim_latents(
            pipe=self.pipe,
            image=batch['source_image'],
            text_embbeding=source_embeds,
            num_inference_steps=self.num_inversion_steps,
            generator=torch.Generator().manual_seed(self.seed),
            controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
            guidance_scale=self.ddim_guidance_scale,
        )

        # if dataset is not list, convert to list
        for key in ['target_ldr_envmap', 'target_norm_envmap', 'target_image', 'target_sh_coeffs', 'word_name']:
            if key in batch and not isinstance(batch[key], list):
                batch[key] = [batch[key]]

        if USE_LIGHT_DIRECTION_CONDITION:
            #Apply the target light direction
            self.select_batch_keyword(batch, 'target')    

        if USE_FROM_PROMPT_GENERATION:
            # generate from text prompt to make sure model is not collapse (yet)
            sd_args = {
                "prompt": batch['text'],
                "output_type": "pt",
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed),
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch, array_index=target_idx)  

            # we store previous scheduler (likely DDIM) and switch to scheduler for regular generation (likely UniPC)
            prev_scheduler = self.pipe.scheduler 
            self.pipe.scheduler = self.regular_scheduler

            sd_image, _ = self.pipe(**sd_args)

            # reverse back to previous scheduler after generation
            self.pipe.scheduler = prev_scheduler

            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}"
                # save image file
                os.makedirs(f"{log_dir}/{epoch_text}from_prompt", exist_ok=True)
                torchvision.utils.save_image(sd_image, f"{log_dir}/{epoch_text}from_prompt/{filename}.png")




        # compute inpaint from nozie
        mse_output = []
        for target_idx in range(len(batch['target_image'])):
            # generate image from DDIM inversion 
            target_light_features = self.get_light_features(batch, array_index=target_idx, generator=torch.Generator().manual_seed(self.seed))            
            target_light_embeds =  self.light2token(source_light_features)
            target_embeds = torch.cat([prompt_embeds, target_light_embeds], dim=-2)
            target_negative_prompt_embeds = torch.cat([prompt_embeds, source_light_embeds], dim=-2)
            pipe_args = {
                "prompt_embeds": target_embeds,
                "negative_prompt_embeds": target_negative_prompt_embeds,
                "output_type": "pt",
                "guidance_scale": self.guidance_scale,
                "return_dict": False,
                "num_inference_steps": self.num_inversion_steps,
                "generator": torch.Generator().manual_seed(self.seed),
                "latents": ddim_latents[-1]
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch, array_index=target_idx)  

            pt_image, _ = self.pipe(**pipe_args)
           
            # generation only to see if everything still work as expected
            sd_args = {
                "prompt_embeds": target_embeds,
                "negative_prompt_embeds": target_negative_prompt_embeds,
                "output_type": "pt",
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed),
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch, array_index=target_idx)  

            # we store previous scheduler (likely DDIM) and switch to scheduler for regular generation (likely UniPC)
            prev_scheduler = self.pipe.scheduler 
            self.pipe.scheduler = self.regular_scheduler

            sd_image, _ = self.pipe(**sd_args)

            # reverse back to previous scheduler after generation
            self.pipe.scheduler = prev_scheduler

           
            # concate image with ground truth
            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0 #bump back to range[0-1]

            gt_image = gt_image.to(pt_image.device)
            tb_image = [gt_image, pt_image, sd_image]
            
            images = torch.cat(tb_image, dim=0)
            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
            
            tb_name = f"{batch['name'][0].replace('/','-')}/{batch['word_name'][target_idx][0].replace('/','-')}"
            
            # calcuarte psnr 
            mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
            mse_output.append(mse[None])
            psnr = -10 * torch.log10(mse)
            
            self.logger.experiment.add_image(f'{tb_name}', image, self.global_step)
            self.log('psnr', psnr)
            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"

                # save with ground truth
                os.makedirs(f"{log_dir}/{epoch_text}with_groudtruth", exist_ok=True)
                torchvision.utils.save_image(image, f"{log_dir}/{epoch_text}with_groudtruth/{filename}.jpg")

                # save image file
                os.makedirs(f"{log_dir}/{epoch_text}crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{log_dir}/{epoch_text}crop_image/{filename}.png")

                # save image file
                os.makedirs(f"{log_dir}/{epoch_text}only_generation", exist_ok=True)
                torchvision.utils.save_image(sd_image, f"{log_dir}/{epoch_text}only_generation/{filename}.png")

                # save psnr to file
                os.makedirs(f"{log_dir}/{epoch_text}psnr", exist_ok=True)
                with open(f"{log_dir}/{epoch_text}psnr/{filename}.txt", "w") as f:
                    f.write(f"{psnr.item()}\n")

                # save prompt
                os.makedirs(f"{log_dir}/{epoch_text}prompt", exist_ok=True) 
                with open(f"{log_dir}/{epoch_text}prompt/{filename}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{log_dir}/{epoch_text}source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{log_dir}/{epoch_text}source_image/{filename}.png")
            if True:              
                if self.global_step == 0:
                    self.logger.experiment.add_text(f'text/{batch["word_name"][0]}', batch['text'][0], self.global_step)
                if self.global_step == 0 and batch_idx == 0:
                    self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
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
            {'params': self.trainable_kv, 'lr': self.learning_rate},
            {'params': self.light2token.parameters(), 'lr': self.learning_rate},
        ])
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale

def set_circle_in_tensor(tensor):
    """
    Modifies a PyTorch tensor of shape [n, 3, 512, 512].
    Sets values inside a circle of size 128x128 at the center of each 2D plane to 1,
    keeping values outside the circle unchanged.

    Args:
        tensor (torch.Tensor): Input tensor of shape [n, 3, 512, 512].

    Returns:
        torch.Tensor: Modified tensor with the circle applied.
    """
    assert tensor.ndim == 4, "Input tensor must be 4D (n, 3, 512, 512)."
    assert tensor.shape[1:] == (3, 512, 512), "Tensor shape must be [n, 3, 512, 512]."
    
    n, _, h, w = tensor.shape
    center_x, center_y = w // 2, h // 2
    radius = 128 // 2

    # Create a grid of indices
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist_from_center = (x - center_x) ** 2 + (y - center_y) ** 2

    # Create a mask for the circle
    circle_mask = dist_from_center <= radius ** 2  # Boolean mask for the circle

    # Apply the mask to each image in the tensor
    circle_mask = circle_mask.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 512, 512]
    circle_mask = circle_mask.expand(n, 3, h, w)  # Match the input tensor shape

    # Set values inside the circle to 1
    tensor[circle_mask] = 1

    return tensor