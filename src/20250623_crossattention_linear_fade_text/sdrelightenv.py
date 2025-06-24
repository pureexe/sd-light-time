"""
Diffusion Renderer reimplmenet
"""

import torch
import torchvision
import lightning as L
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, StableDiffusionPipeline, DDPMScheduler, UNet2DModel, DDIMInverseScheduler, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from LightAttentionProcessor import LightAttentionProcessor
from peft import LoraConfig
import os 
import bitsandbytes 
from env_encoder import EnvEncoder
from types import MethodType
MASTER_TYPE = torch.float16

class SDRelightEnv(L.LightningModule):

    def __init__(
            self,
            lr=1e-4,
            mul_lr_gate=1.0,
            num_conv_in_channels=4 * 4, # latent, albedo, normal, depth
            lora_rank=256, 
            fade_step=50000,
            *args,
            **kwargs
        ):
        super().__init__()
        self.fade_step = fade_step
        self.num_conv_in_channels = num_conv_in_channels
        self.lora_rank = lora_rank
        self.learning_rate = lr
        self.mul_lr_gate = mul_lr_gate
        self.num_inference_step = 500
        self.save_hyperparameters()
        self.is_plot_train_loss = True
        self.seed = 42
        self.lr_expo_decay = 1.0
        self.guidance_scale = 1.0
        self.setup_sd()

    def setup_lora(self):
        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.pipe.unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.lora_params = torch.nn.ParameterList(lora_layers)

    def setup_conv_in(self):
        unet = self.unet
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

    def setup_trainable(self):
        pass        

    def setup_light_feature_projection(self, in_channels=[320, 640, 1280, 1280], out_channels=768):
        # setup CNN to project light
        self.light_feature_projection = torch.nn.ModuleList()
        for in_channel in in_channels:
            linear = torch.nn.Linear(
                in_features=in_channel,
                out_features=out_channels,
            )
            self.light_feature_projection.append(linear)

        self.light_feature_projection.train()

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

        # need to set scheduler to DDIMScheduler for reconstruct from DDIM
        self.normal_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        
        # Patch the step method
        normal_scheduler_step = self.normal_scheduler.step
        def normal_scheduler_patched_step(self, model_output, timestep, sample, *args, **kwargs):
            sample = sample[:, :4] # Slice `sample` (which is usually the `latents`) to [:, :4]
            return normal_scheduler_step(model_output, timestep, sample, *args, **kwargs)
        self.normal_scheduler.step = MethodType(normal_scheduler_patched_step, self.normal_scheduler)

        # Patch the step method
        inverse_scheduler_step = self.inverse_scheduler.step
        def inverse_scheduler_patched_step(self, model_output, timestep, sample, *args, **kwargs):
            sample = sample[:, :4] # Slice `sample` (which is usually the `latents`) to [:, :4]
            return inverse_scheduler_step(model_output, timestep, sample, *args, **kwargs)
        self.inverse_scheduler.step = MethodType(inverse_scheduler_patched_step, self.inverse_scheduler)

        self.pipe.scheduler = self.normal_scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)


        self.setup_lora()
        self.setup_light_feature_projection()

        # setup trainable
        self.setup_trainable()

        # setup attention related
        self.setup_attention_processor()

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

    def setup_attention_processor(
        self,
        token_lengths = [4096, 1024, 256, 64] # number of tokens for each depth
    ):
        # setup attention processor
        attention_dict = {}
        default_attn_processor = AttnProcessor2_0()
        
        # setup attention gate 
        attention_light_gate = torch.zeros(768)

        processor_id = 0
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
                attn_processor = LightAttentionProcessor(processor_id=processor_id, block_depth=depth, token_lengths=token_lengths)
                attention_light_gate[processor_id * 2] = 1.0
                processor_id += 1
                attention_dict[attn_name] = attn_processor
        self.attention_light_gate_total = processor_id
        self.pipe.unet.set_attn_processor(attention_dict)
        self.attention_light_gate = attention_light_gate


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
        light_features = []
        for layer_id in range(4):
            light_features.append(
                batch[f'envmap_feature_layer{layer_id}']
            )
        return light_features
    
    def get_light_embedings(self, light_features):
        """
        Get light embedings from light features
        """
        # concatenate light embedings along channel dimension
        light_out_embedings = []
        for i, embed in enumerate(light_features): 
            # embeding [B, C, H, W]
            # flatten to [B, C, H*W] and transpose to [B, H*W, C]
            embed = embed.flatten(2).transpose(1, 2)  # [B, Token, 320]
            out_embed = self.light_feature_projection[i](embed) # [B, Token, 768]
            light_out_embedings.append(out_embed)  # append to list

        # Combine
        light_embedings = torch.cat(light_out_embedings, dim=1)  # [B, seq_len*4, 768]
        return light_embedings

    
    def get_spatial_features(self, batch, latents):
        # in case network accept more than latents
        return latents

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

        # compute text embedding 
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt = batch['text'],
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )

        # get timesteps for training 
        timesteps = self.get_timesteps(timesteps, batch_size, device) # range [0, 1000] for training

        # get light features       
        light_features = self.get_light_features(batch)
        light_embedings = self.get_light_embedings(light_features)  # [B, C*3, H, W]

        query_embedings = self.get_query_embedings(
            text_embedings=prompt_embeds, 
            light_embedings=light_embedings
        )
 
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
            encoder_hidden_states=query_embedings,
            return_dict=False
        )[0]

        # compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, target)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def update_attention_gate(self):
        current_ratio = self.global_step / self.fade_step
        if current_ratio > 1.0:
            current_ratio = 1.0
        for processor_id in range(self.attention_light_gate_total):
            self.attention_light_gate[processor_id * 2] = 1.0 - current_ratio 
            self.attention_light_gate[processor_id * 2 + 1] = current_ratio
            
    
    def get_query_embedings(self, text_embedings,light_embedings):
        """
        Get query embedings for the attention processor
        """
        # concatenate text and light embedings
        batch_size = light_embedings.shape[0]

        self.update_attention_gate()

        device = text_embedings.device

        query_embedings = torch.cat([
            text_embedings,  # B, 77, 768
            self.attention_light_gate[None, None].expand(batch_size, -1, -1).to(device),
            light_embedings
        ], dim=1)
        return query_embedings
    
    def generate_images(self, batch, batch_idx, noisy_latents=None, num_inference_step=None, prompt_embeds=None):
        device = batch['image'].device
        if num_inference_step is None:
            num_inference_step = self.num_inference_step

        # compute feature for first loop
        if noisy_latents is None:
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

        # compute text embedding 
        if prompt_embeds is None:
            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt = batch['text'],
                device = device,
                num_images_per_prompt=1, 
                do_classifier_free_guidance=False
            )

        light_embedings = self.get_light_embedings(light_features)

        # get query embedings for attention processor
        query_embedings = self.get_query_embedings(
            text_embedings=prompt_embeds, 
            light_embedings=light_embedings
        )

        # compute feature for future loop
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            # in last step, we don't need to mess up with latent
            if i >= num_inference_step - 1:
                return callback_kwargs
           
            # concatenate latents for next step
            spatial_features = self.get_spatial_features(batch, callback_kwargs['latents'])
            callback_kwargs['latents'] = spatial_features

            return callback_kwargs

        pipe_args = {
            "latents": spatial_features,  # [B, C*4, H, W]
            "prompt_embeds": query_embedings,  # [B, C*3, H, W]
            "negative_prompt_embeds": query_embedings,  # [B, C*3, H, W]
            "output_type": "pt",
            "guidance_scale": self.guidance_scale,
            "return_dict": False,
            "num_inference_steps": num_inference_step,
            "generator": torch.Generator().manual_seed(self.seed),
            "callback_on_step_end": callback_on_step_end,
            "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds", "negative_prompt_embeds"],
            "light_features": light_features,  # [B, C*3, H, W]
            "spatial_features": spatial_features,  # [B, C*3, H, W]
        }

        pt_image, _ = self.pipe(**pipe_args)
        return pt_image
    
    def get_ddim_latents(self, batch, batch_idx, prompt_embeds=None):

        device = batch['image'].device
        
        # swap the scheduler to the inverse 
        self.pipe.scheduler = self.inverse_scheduler

        # compute feature for first loop
        sharp_latents = self.get_latents_from_images(batch['image'])

        light_features = self.get_light_features(batch)
        spatial_features = self.get_spatial_features(batch, sharp_latents)
        light_embedings = self.get_light_embedings(light_features)

        # compute text embedding 
        if prompt_embeds is None:
            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt = batch['text'],
                device = device,
                num_images_per_prompt=1, 
                do_classifier_free_guidance=False
            )

        query_embedings = self.get_query_embedings(
            text_embedings=prompt_embeds, 
            light_embedings=light_embedings
        )
        
        ddim_latents = []
        ddim_timesteps = []
    
        def callback_ddim_on_step_end(pipe, i, t, callback_kwargs):
            ddim_timesteps.append(t)
            ddim_latents.append(callback_kwargs['latents'])

            # in last step, we don't need to mess up with latent
            if i >= self.num_inference_step - 1: # don't update last step
                return callback_kwargs
            # compute light features for next step
            try:
                # concatenate latents for next step
                spatial_features = self.get_spatial_features(batch, callback_kwargs['latents'])
                callback_kwargs['latents'] = spatial_features
            except:
                pass

            return callback_kwargs
                
        pipe_args = {
            "latents": spatial_features,  # [B, C*4, H, W]
            "prompt_embeds": query_embedings,  # [B, C*3, H, W]
            "negative_prompt_embeds": query_embedings,  # [B, C*3, H, W]
            "output_type": "pt",
            "guidance_scale": self.guidance_scale,
            "return_dict": False,
            "num_inference_steps": self.num_inference_step,
            "generator": torch.Generator().manual_seed(self.seed),
            "callback_on_step_end": callback_ddim_on_step_end,
            "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds", "negative_prompt_embeds"],
        }
        
        zt_noise, _ = self.pipe(**pipe_args)

        # swap the scheduler back 
        self.pipe.scheduler = self.normal_scheduler
        return ddim_latents[-1]
    
    def set_num_inference_step(self, num_inference_step):
        """
        Set the number of inference step for the model
        """
        self.num_inference_step = num_inference_step

    def select_batch(self, batch, prefix='source', index=0):
        LIGHT_KEYS = ['light_ldr', 'light_log_hdr', 'light_dir', 'irradiant_ldr', 'irradiant_log_hdr', 'irradiant_dir', 'image', 'envmap_feature_layer0', 'envmap_feature_layer1', 'envmap_feature_layer2', 'envmap_feature_layer3']
        for key in LIGHT_KEYS:
            if not prefix + '_' + key in batch:
                continue
            if prefix == 'target':
                if isinstance(batch[prefix + '_' + key], list):
                    batch[key] = batch[prefix + '_' + key][index]
                else:
                    batch[key] = batch[prefix + '_' + key]
            else:
                batch[key] = batch[prefix + '_' + key]
        
        return batch


    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):   
        # check key first
        log_dir = self.get_logdir()

        device = batch['source_image'].device

        # compute text embedding, we compute here so when doing inversion and generation
        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt = batch['text'],
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )

        epoch_text = f"step_{self.global_step:06d}/" if is_seperate_dir_with_epoch else ""
        source_name = batch['name'][0].replace('/','-')

        batch = self.select_batch(batch, prefix='source')
        noisy_latents = self.get_ddim_latents(batch, batch_idx, prompt_embeds) # range [-1, 1]

        for light_id in range(len(batch['envmap_name'])):
            target_name = batch['envmap_name'][light_id][0].replace('/','-')
            filename = f"{source_name}_{target_name}"

            batch = self.select_batch(batch, prefix=f'target', index=light_id)

            pt_image = self.generate_images(batch, batch_idx, noisy_latents=noisy_latents, prompt_embeds=prompt_embeds) # range [0, 1]
            gt_image = batch['image'] # range [-1,1]
            gt_image = (gt_image + 1.0) / 2.0  # convert to [0, 1] range
            
            # generate image without inversion for comparision
            sd_image = self.generate_images(batch, batch_idx, num_inference_step=50, prompt_embeds=prompt_embeds) # range [0, 1]

            tb_image = [gt_image, pt_image, sd_image]
            images = torch.cat(tb_image, dim=0)
            images = torch.clamp(images, 0.0, 1.0)

            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)

            # calcuarte psnr 
            mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
            psnr = -10 * torch.log10(mse)

            self.log(f'psnr/{filename}', psnr)
            self.log(f'mse/{filename}', mse)
            
            if is_save_image:
            
                # save with ground truth
                os.makedirs(f"{log_dir}/{epoch_text}with_groudtruth", exist_ok=True)
                print(f"Predicted to: {log_dir}/{epoch_text}with_groudtruth")
                torchvision.utils.save_image(image, f"{log_dir}/{epoch_text}with_groudtruth/{filename}.jpg")
                os.chmod(f"{log_dir}/{epoch_text}with_groudtruth", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}with_groudtruth/{filename}.jpg", 0o777)

                # save image file
                os.makedirs(f"{log_dir}/{epoch_text}crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{log_dir}/{epoch_text}crop_image/{filename}.png")
                os.chmod(f"{log_dir}/{epoch_text}crop_image", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}crop_image/{filename}.png", 0o777)

                # save sd image file
                os.makedirs(f"{log_dir}/{epoch_text}sd_image", exist_ok=True)
                torchvision.utils.save_image(sd_image, f"{log_dir}/{epoch_text}sd_image/{filename}.png")
                os.chmod(f"{log_dir}/{epoch_text}sd_image", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}sd_image/{filename}.png", 0o777)

                # save control image to verify everything is correct 
                is_save_control_image = True
                if is_save_control_image:
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
        self.generate_tensorboard(batch, batch_idx, is_save_image=True)
        # if self.is_plot_train_loss:
        #     self.plot_train_loss(batch, batch_idx, is_save_image=True, seed=self.seed)
        # else:
        #     self.generate_tensorboard(batch, batch_idx, is_save_image=True)

    #TODO: let's create seperate file that take checkpoint and compute the loss. this loss code should be re-implememet
    def plot_train_loss(self, batch, batch_idx, is_save_image=False, seed=None):
        self.select_batch(batch, prefix=f'source')
        for timestep in range(100, 1000, 100):
            loss = self.compute_train_loss(batch, batch_idx, timesteps=timestep, seed=seed)
            self.logger.experiment.add_scalar(f'plot_train_loss/{timestep}', loss, self.global_step)
            self.logger.experiment.add_scalar(f'plot_train_loss/average', loss, self.global_step)
            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}"
                os.makedirs(f"{self.logger.log_dir}/train_loss/{timestep}", exist_ok=True)
                with open(f"{self.logger.log_dir}/train_loss/{timestep}/{filename}.txt", "w") as f:
                    f.write(f"{loss.item()}")
        # plot light_gate 
        for i in range(self.attention_light_gate_total):
            self.logger.experiment.add_scalar(f'gate_text/{i:02d}', self.attention_light_gate[i*2].item(), self.global_step)
            self.logger.experiment.add_scalar(f'gate_light/{i:02d}', self.attention_light_gate[i*2+1].item(), self.global_step)


    def validation_step(self, batch, batch_idx):
        self.plot_train_loss(batch, batch_idx, seed=None) # DEFAULT USE SEED 42
        mse = self.generate_tensorboard(batch, batch_idx, is_save_image=True, is_seperate_dir_with_epoch=True) #NOTE: temporary enable save image for predict the light
        return mse
    
    def configure_optimizers(self):
        optimizer_class = bitsandbytes.optim.AdamW8bit # bitsandbytes is used for 8-bit AdamW optimizer
        # torch.optim.Adam

        optimizer = optimizer_class([
            {'params': self.lora_params, 'lr': self.learning_rate},
            {'params': self.light_feature_projection.parameters(), 'lr': self.learning_rate}  # Light feature projection parameters
        ])

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


class SDNormalDepthRelightEnv(SDRelightEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_conv_in_channels = 4 * 3

    def setup_trainable(self):
        super().setup_trainable()
        # setup conv_in
        in_channels = 4 * 3  # latent, normal, depth
        new_conv = expand_conv_in_channels(self.pipe.unet.conv_in, new_in_channels=in_channels)
        self.pipe.unet.conv_in = new_conv
        self.pipe.unet.conv_in.requires_grad_(True)  # make conv_in trainable

        # we force the conv_in to be trainable in LoRA parameters. might be fix in the future
        lora_layers = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.lora_params = torch.nn.ParameterList(lora_layers)


    def get_spatial_features(self, batch, latents):
        """
        Get spatial features for the input
        """
        if latents is not None:
            output_features = [latents]
        else:
            output_features = []
        for feature in ['normal', 'depth']:
            if feature in batch:
                feature_tensor = batch[feature]
                if feature_tensor is not None:
                    feature_tensor = self.get_latents_from_images(feature_tensor)
                    output_features.append(feature_tensor)
        output_features = torch.cat(output_features, dim=1)  # [B, C*3, H, W]
        return output_features



class SDAlbedoNormalDepthRelightEnv(SDRelightEnv):
    def setup_trainable(self):
        super().setup_trainable()
        # setup conv_in
        in_channels = 4 * 4  # latent, albedo, normal, depth
        new_conv = expand_conv_in_channels(self.pipe.unet.conv_in, new_in_channels=in_channels)
        self.pipe.unet.conv_in = new_conv
        self.pipe.unet.conv_in.requires_grad_(True)  # make conv_in trainable

        # we force the conv_in to be trainable in LoRA parameters. might be fix in the future
        lora_layers = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.lora_params = torch.nn.ParameterList(lora_layers)


    def get_spatial_features(self, batch, latents):
        """
        Get spatial features for the input
        """
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

class SDRelightIrradientEnv(SDRelightEnv):

    def setup_attention_processor(
        self,
        token_lengths = [4096, 1024, 256, 64] # number of tokens for each depth
    ):
        for i in range(len(token_lengths)):
            token_lengths[i] = token_lengths[i] * 2 # one for envmap other for irradient
        # setup attention processor
        super().setup_attention_processor(token_lengths=token_lengths)

    def get_light_features(self, batch):
        """
        Encode the light using light encoder
        """
        light_ldr = self.get_svdvae_latents_from_images(batch['light_ldr'])
        light_log_hdr = self.get_svdvae_latents_from_images(batch['light_log_hdr'])
        light_dir = self.get_svdvae_latents_from_images(batch['light_dir'])
        irradiant_ldr = self.get_svdvae_latents_from_images(batch['irradiant_ldr'])
        irradiant_log_hdr = self.get_svdvae_latents_from_images(batch['irradiant_log_hdr'])
        irradiant_dir = self.get_svdvae_latents_from_images(batch['irradiant_dir'])

        features_light = [
            light_ldr,  # [B, C, H, W]
            light_log_hdr,  # [B, C, H, W]
            light_dir  # [B, C, H, W]            
        ]        
        features_light = torch.cat(features_light, dim=1)  # [B, C*3, H, W]

        features_irradiant = [
            irradiant_ldr,  # [B, C, H, W]
            irradiant_log_hdr,  # [B, C, H, W]
            irradiant_dir  # [B, C, H, W]            
        ]
        features_irradient = torch.cat(features_irradiant, dim=1)  # [B, C*3, H, W]
        features = [features_light, features_irradient]
        return features
    
    def get_light_embedings(self, features):
        """
        Get light embedings from light features
        """
        light_embedings = self.encoder['env'](features[0]) # return list with shape [b, 320,1040, H, W]
        irradient_embedings = self.encoder['env'](features[1]) # return list with shape [b, 320,1040, H, W]
        # concatenate light embedings along channel dimension
        light_out_embedings = []
        for i in range(len(light_embedings)):
            light = light_embedings[i].flatten(2).transpose(1, 2)  # [B, Token, 320]
            irradient = irradient_embedings[i].flatten(2).transpose(1, 2)  # [B, Token, 320]
            out_embed = torch.cat([light, irradient], dim=-2)  # [B, Token*2, 320]
            out_embed = self.light_feature_projection[i](out_embed) # [B, Token*2, 768]
            light_out_embedings.append(out_embed)  # append to list

        # Combine
        light_embedings = torch.cat(light_out_embedings, dim=1)  # [B, seq_len*4, 768]
        return light_embedings
    
class SDAlbedoNormalDepthRelightIrradientEnv(SDRelightIrradientEnv,SDAlbedoNormalDepthRelightEnv):
    def place_holders(self):
        """
        This is a placeholder for the class to ensure it can be instantiated.
        It inherits from both SDRelightIrradientEnv and SDAlbedoNormalDepthRelightEnv.
        """
        pass
    
def expand_conv_in_channels(old_conv, new_in_channels):

    # Example use:
    # old_conv = nn.Conv2d(4, 64, kernel_size=3, padding=1)
    # new_conv = expand_conv_in_channels(old_conv, new_in_channels=12)


    # Sanity check
    assert new_in_channels >= old_conv.in_channels, "New in_channels must be >= old in_channels"

    # Create new Conv2d layer
    new_conv = torch.nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode
    )

    # Copy existing weights
    with torch.no_grad():
        new_conv.weight[:, :old_conv.in_channels, :, :] = old_conv.weight
        new_conv.weight[:, old_conv.in_channels:, :, :] = 0.0  # Zero the new channels

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    return new_conv

