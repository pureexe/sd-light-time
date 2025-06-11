"""
Diffusion Renderer reimplmenet
"""

import torch
import torchvision
import lightning as L
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, StableDiffusionPipeline, DDPMScheduler, UNet2DModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from LightEncoder import LightEncoder
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from LightAttentionProcessor import LightAttentionProcessor
from peft import LoraConfig
import os 
import bitsandbytes 
from StableDiffusionRelightPipeline import StableDiffusionRelightPipeline
from types import MethodType

MASTER_TYPE = torch.float16

class SDDiffusionRenderer(L.LightningModule):

    def __init__(
            self,
            lr_conv_in=1e-4,
            lr_light_encoder=1e-4,
            lr_lora=1e-4,
            num_conv_in_channels=4 * 4, # latent, albedo, normal, depth
            lora_rank=256,
            *args,
            **kwargs
        ):
        super().__init__()
        self.num_conv_in_channels = num_conv_in_channels
        self.lora_rank = lora_rank
        self.lr_light_encoder = lr_light_encoder
        self.lr_conv_in = lr_conv_in
        self.lr_lora = lr_lora
        self.num_infernece_step = 50
        self.save_hyperparameters()
        self.is_plot_train_loss = True
        self.seed = 42
        self.lr_expo_decay = 1.0
        self.guidance_scale = 1.0
        self.setup_sd()

    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5"):
        # load unet
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')
        # load pipeline
        self.pipe =  StableDiffusionPipeline.from_pretrained(
            sd_path,
            unet = unet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        ).to('cuda')

        original_step = self.pipe.scheduler.step
        # Define wrapper
        def patched_step(self, model_output, timestep, sample, *args, **kwargs):
            # Slice `sample` (which is usually the `latents`) to [:, :4]
            sample = sample[:, :4]
            return original_step(model_output, timestep, sample, *args, **kwargs)
        # Patch the method
        self.pipe.scheduler.step = MethodType(patched_step, self.pipe.scheduler)



        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

        # Light encoder
        
        self.light_encoder = LightEncoder()

        # setup trainable
        self.lora_params = torch.nn.ParameterList(lora_layers)
        
        # first conv layer 
        conv_in = torch.nn.Conv2d(
            in_channels=self.num_conv_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            bias=False
        )
        self.pipe.unet.conv_in = conv_in
        self.conv_in = self.pipe.unet.conv_in

        # setup attention processor
        attention_dict = {}
        default_attn_processor = AttnProcessor2_0()
        token_lengths = [256, 64, 16, 16]
        for attn_name in self.pipe.unet.attn_processors.keys():
            if attn_name.endswith("attn1.processor"): # first attention is self attention
                attention_dict[attn_name] = default_attn_processor
            else: # second attention is cross attention
                attention_dict[attn_name] = LightAttentionProcessor()
                names = attn_name.split('.')
                if names[0] == 'down_blocks':
                    depth = int(names[1])
                if names[0] == 'mid_block':
                    depth = 3
                if names[0] == 'up_blocks':
                    # unlike down_blocks, up_blocks are reversed but start with index-1 instead of 0
                    depth = 3 - int(names[1])
                attn_processor = LightAttentionProcessor(block_depth=depth, token_lengths=token_lengths)
                attention_dict[attn_name] = attn_processor
        self.pipe.unet.set_attn_processor(attention_dict)


        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

    def get_logdir(self):
        try:
            log_dir = self.logger.log_dir
        except:
            log_dir = self.log_dir
        try:
            os.chmod(log_dir, 0o777)
        except:
            pass
        return log_dir
    
    def get_vae_shiftscale(self):
        shift = self.pipe.vae.config.shift_factor if hasattr(self.pipe.vae.config, 'shift_factor') else 0.0
        scale = self.pipe.vae.config.scaling_factor if hasattr(self.pipe.vae.config, 'scaling_factor') else 1.0
        # clear out None type 
        shift = 0.0 if shift is None else shift
        scale = 1.0 if scale is None else scale
        return shift, scale
    
    def get_latents_from_images(self, image):
        """
        Encode the image using VAE
        """
        with torch.no_grad():
            latents = self.pipe.vae.encode(image).latent_dist.sample().detach()
            shift, scale = self.get_vae_shiftscale()
            latents = (latents - shift) * scale       
        return latents
    
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

    def get_light_features(self, batch):
        """
        Encode the light using light encoder
        """
        light_ldr = self.get_latents_from_images(batch['light_ldr'])
        light_log_hdr = self.get_latents_from_images(batch['light_log_hdr'])
        light_dir = self.get_latents_from_images(batch['light_dir'])

        features = [
            light_ldr,  # [B, C, H, W]
            light_log_hdr,  # [B, C, H, W]
            light_dir  # [B, C, H, W]
        ]
        features = torch.cat(features, dim=1)  # [B, C*3, H, W]
        return features
    
    def get_spatial_features(self, batch, latents):
        """
        Get spatial features for the input
        """
        # batch['image'], batch['albedo'], batch['normal'], batch['depth']
        if latents is not None:
            output_features = [latents]
        else:
            output_features = []
        for feature in ['albedo', 'normal', 'depth']:
            if feature in batch:
                feature_tensor = batch[feature]
                if feature_tensor is not None:
                    feature_tensor = self.get_latents_from_images(feature_tensor)
                    output_features.append(feature_tensor)
        output_features = torch.cat(output_features, dim=1)  # [B, C*4, H, W]
        return output_features

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

    def compute_train_loss(self, batch, batch_idx, timesteps = None, seed = None):

        device = batch['image'].device
        batch_size = batch['image'].shape[0]

        # get timesteps for training 
        timesteps = self.get_timesteps(timesteps, batch_size, device) # range [0, 1000] for training

        # get light features       
        light_features = self.get_light_features(batch)
        light_embedings = self.light_encoder(light_features, timestep=timesteps, return_dict=False)
 
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # compute image original latent
        latents = self.get_latents_from_images(batch['image'])

        # get gaussain noise 
        noise = self.get_noise_from_latents(latents)
        target = noise # this noise also be a training target for our model


        # add noise from a clear latents to specific corrupt noise that have match timestep
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)       

        # prepare input
        spatial_features = self.get_spatial_features(batch, noisy_latents)

        # forward unet
        noise_pred = self.pipe.unet(
            spatial_features,
            timestep=timesteps,
            encoder_hidden_states=light_embedings,
            return_dict=False
        )[0]

        # compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, target)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def generate_images(self, batch, batch_idx):
        device = batch['image'].device
        # get light features 
        timesteps, num_inference_step = retrieve_timesteps(
            self.pipe.scheduler, self.num_infernece_step, device, None, None
        )

        # compute feature for first loop
        noisy_latents = self.pipe.prepare_latents(
            batch_size=batch['image'].shape[0],
            height=batch['image'].shape[2],
            width=batch['image'].shape[3],
            dtype=MASTER_TYPE,
            device=batch['image'].device,
            generator=torch.Generator().manual_seed(self.seed),
            num_channels_latents=4
        )

        light_features = self.get_light_features(batch)
        spatial_features = self.get_spatial_features(batch, noisy_latents)

        light_embedings = self.light_encoder(light_features, timestep=timesteps[0], return_dict=False)
        # compute feature for future loop
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            # in last step, we don't need to mess up with latent
            if i > num_inference_step - 1:
                return callback_kwargs
            # compute light features for next step
            light_features = self.get_light_features(batch)
            light_embedings = self.light_encoder(light_features, timestep=timesteps[i+1], return_dict=False)
            callback_kwargs['prompt_embeds'] = light_embedings
            callback_kwargs['negative_prompt_embeds'] = light_embedings

            # concatenate latents for next step
            spatial_features = self.get_spatial_features(batch, callback_kwargs['latents'])
            callback_kwargs['latents'] = spatial_features

            return callback_kwargs

        pipe_args = {
            "latents": spatial_features,  # [B, C*4, H, W]
            #"prompt": "a photo realistic image", # temporary place holder, we no longer use prompt
            "prompt_embeds": light_embedings,  # [B, C*3, H, W]
            "negative_prompt_embeds": light_embedings,  # [B, C*3, H, W]
            "output_type": "pt",
            "guidance_scale": self.guidance_scale,
            "return_dict": False,
            "num_inference_steps": self.num_infernece_step,
            "generator": torch.Generator().manual_seed(self.seed),
            "callback_on_step_end": callback_on_step_end,
            "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds", "negative_prompt_embeds"],
            "light_features": light_features,  # [B, C*3, H, W]
            "spatial_features": spatial_features,  # [B, C*3, H, W]
            "light_encoder": self.light_encoder,  # LightEncoder instance
        }

        pt_image, _ = self.pipe(**pipe_args)
        return pt_image

    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):   
        log_dir = self.get_logdir()

        epoch_text = f"step_{self.global_step:06d}/" if is_seperate_dir_with_epoch else ""
        source_name = f"{batch['name'][0].replace('/','-')}"
        
        # create callback to attach albedo, depth. normal 
        spatial_features = self.get_spatial_features(batch, self.get_latents_from_images(batch['image']))    

        pt_image = self.generate_images(batch, batch_idx) # range [0, 1]
        gt_image = batch['image'] # range [-1,1]
        gt_image = (gt_image + 1.0) / 2.0  # convert to [0, 1] range
        

        tb_image = [gt_image, pt_image]
        images = torch.cat(tb_image, dim=0)
        images = torch.clamp(images, 0.0, 1.0)

        image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)

        # calcuarte psnr 
        mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
        psnr = -10 * torch.log10(mse)

        self.log(f'psnr/{source_name}', psnr)
        self.log(f'mse/{source_name}', mse)
        
        if is_save_image:
            filename = f"{source_name}"

            # save with ground truth
            os.makedirs(f"{log_dir}/{epoch_text}with_groudtruth", exist_ok=True)
            torchvision.utils.save_image(image, f"{log_dir}/{epoch_text}with_groudtruth/{filename}.jpg")
            os.chmod(f"{log_dir}/{epoch_text}with_groudtruth", 0o777)
            os.chmod(f"{log_dir}/{epoch_text}with_groudtruth/{filename}.jpg", 0o777)

            # save image file
            os.makedirs(f"{log_dir}/{epoch_text}crop_image", exist_ok=True)
            torchvision.utils.save_image(pt_image, f"{log_dir}/{epoch_text}crop_image/{filename}.png")
            os.chmod(f"{log_dir}/{epoch_text}crop_image", 0o777)
            os.chmod(f"{log_dir}/{epoch_text}crop_image/{filename}.png", 0o777)

            # save control image to verify everything is correct 
            if False:
                for feature_name in ['albedo', 'normal', 'depth', 'light_ldr', 'light_log_hdr', 'light_dir']:
                    if feature_name in batch:
                        feature_image = batch[feature_name]
                        if feature_image is not None:
                            feature_image = (feature_image + 1.0) / 2.0
                            os.makedirs(f"{log_dir}/{epoch_text}control_image/{feature_name}", exist_ok=True)
                            torchvision.utils.save_image(
                                feature_image, 
                                f"{log_dir}/{epoch_text}control_image/{feature_name}/{filename}.png"
                            )
                            os.chmod(f"{log_dir}/{epoch_text}control_image/{feature_name}", 0o777)
                            os.chmod(f"{log_dir}/{epoch_text}control_image/{feature_name}/{filename}.png", 0o777)

            # save psnr to file
            os.makedirs(f"{log_dir}/{epoch_text}psnr", exist_ok=True)
            with open(f"{log_dir}/{epoch_text}psnr/{filename}.txt", "w") as f:
                f.write(f"{psnr.item()}\n")
            os.chmod(f"{log_dir}/{epoch_text}psnr", 0o777)
            os.chmod(f"{log_dir}/{epoch_text}psnr/{filename}.txt", 0o777)

    def test_step(self, batch, batch_idx):
        if self.is_plot_train_loss:
            self.plot_train_loss(batch, batch_idx, is_save_image=True, seed=self.seed)
        else:
            self.generate_tensorboard(batch, batch_idx, is_save_image=True)

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
        self.plot_train_loss(batch, batch_idx, seed=None) # DEFAULT USE SEED 42
        mse = self.generate_tensorboard(batch, batch_idx, is_save_image=True, is_seperate_dir_with_epoch=True) #NOTE: temporary enable save image for predict the light
        return mse
    
    def configure_optimizers(self):
        optimizer_class = bitsandbytes.optim.AdamW8bit # bitsandbytes is used for 8-bit AdamW optimizer
        # torch.optim.Adam

        optimizer = optimizer_class([
            {'params': self.light_encoder.parameters(), 'lr': self.lr_light_encoder},
            {'params': self.conv_in.parameters(), 'lr': self.lr_conv_in},
            {'params': self.lora_params.parameters(), 'lr': self.lr_lora},
        ])

        optimizer = optimizer_class(self.parameters(), lr=self.lr_lora)
        if self.lr_expo_decay != 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_expo_decay)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Decay the learning rate at each epoch
                }
            }
        return optimizer




    def render(self, input_data):
        pass
