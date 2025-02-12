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
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, StableDiffusionPipeline, DDPMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline
from transformers import CLIPImageProcessor, CLIPModel
from InversionHelper import get_ddim_latents
from LightEmbedingBlock import set_light_direction, add_light_block
import torchmetrics

MASTER_TYPE = torch.float16

class StableDiffusionImg2ImgNoPrepLatentsPipeline(StableDiffusionImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        return image
    
class StableDiffusionControlNetNoPrepLatentsPipeline(StableDiffusionControlNetImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        return image

class SDDiffusionFace(L.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            guidance_scale=3,
            feature_type="diffusion_face",
            num_inversion_steps=500,
            num_inference_steps=500,
            use_false_shading=False,
            use_triplet_background=False,
            ctrlnet_lr=1,
            *args,
            **kwargs
        ) -> None:
        super().__init__()
        self._lpips_loss = None
        self._ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)  
        USE_LPIPS = True
        if  self._lpips_loss is USE_LPIPS:
            import lpips
            self._lpips_loss = lpips.LPIPS(net='alex') # best forward scores


        self.condition_scale = 1.0
        self.guidance_scale = guidance_scale
        self.ddim_guidance_scale = 1.0
        self.ddim_strength = 0.0
        self.gaussain_strength = 0.0
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.feature_type = feature_type
        self.ctrlnet_lr = ctrlnet_lr

        self.num_inversion_steps = num_inversion_steps
        self.num_inference_steps = num_inference_steps
        self.use_false_shading = use_false_shading
        self.use_triplet_background = use_triplet_background

        self.save_hyperparameters()

        self.already_load_img2img = False

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
        self.setup_clip()
        self.setup_light_block()
        self.setup_trainable()
        self.log_dir = ""

        if self.use_false_shading or self.use_triplet_background:
            # intialize scheduler for get better latents 
            self.scheduler_ddpm = DDPMScheduler.from_config(self.pipe.scheduler.config)
            self.scheduler_ddpm.set_timesteps(1000)
            self.scheduler_ddpm.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to('cuda')

    def setup_ddim_img2img(self):
        if self.already_load_img2img:
            return
        
        if hasattr(self.pipe,'controlnet'):
            self.pipe_img2img = StableDiffusionControlNetNoPrepLatentsPipeline(
                vae = self.pipe.vae,
                text_encoder = self.pipe.text_encoder,
                tokenizer = self.pipe.tokenizer,
                unet = self.pipe.unet,
                controlnet=self.pipe.controlnet,
                scheduler = self.pipe.scheduler,
                safety_checker = self.pipe.safety_checker,
                feature_extractor = self.pipe.feature_extractor,
                image_encoder = self.pipe.image_encoder
            )
        else:
            self.pipe_img2img = StableDiffusionImg2ImgNoPrepLatentsPipeline(
                vae = self.pipe.vae,
                text_encoder = self.pipe.text_encoder,
                tokenizer = self.pipe.tokenizer,
                unet = self.pipe.unet,
                scheduler = self.pipe.scheduler,
                safety_checker = self.pipe.safety_checker,
                feature_extractor = self.pipe.feature_extractor,
                image_encoder = self.pipe.image_encoder
            )
        self.already_load_img2img = True

    def setup_clip(self, clip_model = "openai/clip-vit-large-patch14"):
        if self.feature_type not in ['clip_shcoeff', 'clip']:
            return None 
        # load clip model
        with torch.inference_mode():
            self.clip_features = {
                'model': CLIPModel.from_pretrained(clip_model, torch_dtype=MASTER_TYPE).to('cuda'),
                'processor': CLIPImageProcessor.from_pretrained(clip_model)
            }
        


    def setup_trainable(self):
        # register trainable module to pytorch lighting model 

        # register adaptive group_norm
        adaptive_group_norm = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.adaptive_group_norm = torch.nn.ParameterList(adaptive_group_norm)

        # register controlnet 
        self.controlnet_trainable = torch.nn.ParameterList(self.pipe.controlnet.parameters())


    def setup_light_block(self):
        if self.feature_type == "diffusion_face":
            mlp_in_channel = 616
        elif self.feature_type == "diffusion_face_shcoeff":
            mlp_in_channel = 616 + 27
        elif self.feature_type == "clip_shcoeff":
            mlp_in_channel = 768 + 27
        elif self.feature_type == "shcoeff_order2":
            mlp_in_channel = 27
        elif self.feature_type == "clip":
            mlp_in_channel = 768
        elif self.feature_type == "vae":
            mlp_in_channel = 4 * 32 * 32 # 4096
        elif self.feature_type == "vae_shcoeff":
            mlp_in_channel = (4 * 32 * 32) + 27# 4096
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')


    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # load main UNet
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')

        # create controlnet from unet. Unet take 6 channel input with are 3 channel of background and other 3 channel of shading
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6) 

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            unet=unet, # We add unet here to prevent it from reloading Unet
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

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
        if self.feature_type in ["diffusion_face", "diffusion_face_shcoeff", "shcoeff_order2"]:
            diffusion_face = batch['diffusion_face']
            if array_index is not None:
                diffusion_face = diffusion_face[array_index]
            diffusion_face = diffusion_face.view(diffusion_face.size(0), -1) #flatten shcoeff to [B, 616]
            return diffusion_face
        elif self.feature_type in ['clip_shcoeff', 'clip']:
            diffusion_face = batch['diffusion_face']
            if self.feature_type in ['clip_shcoeff']:
                if array_index is not None:
                    sh_coeff = diffusion_face[array_index][..., -27:]
                else:
                    sh_coeff = diffusion_face[..., -27:]
            
            source_image = batch['source_image']

            # Ensure input is in the expected format
            if source_image.ndim != 4 or source_image.shape[1:] != (3, 512, 512):
                raise ValueError("Input tensor must have shape [batch, 3, 512, 512].")
            if source_image.min().item() < -1 or source_image.max().item() > 1:
                raise ValueError("Input tensor values must be in the range [-1, 1].")
                        
            # Convert range [-1, 1] to [0, 1] for CLIP
            source_image = (source_image + 1) / 2
            
            # Process images
            inputs = self.clip_features['processor'](images=source_image, return_tensors="pt").to('cuda').to(MASTER_TYPE)
            
            # Extract image embeddings
            with torch.no_grad():
                clip_features = self.clip_features['model'].get_image_features(pixel_values=inputs['pixel_values'])  # Shape [batch, 768]
            if self.feature_type in ['clip_shcoeff']:
                features = torch.cat([clip_features, sh_coeff], dim=-1)
            else:
                features = clip_features
            return features
        elif self.feature_type in ['vae_shcoeff', 'vae']:
            diffusion_face = batch['diffusion_face']
            if self.feature_type in ['vae_shcoeff']:
                if array_index is not None:
                    sh_coeff = diffusion_face[array_index][..., -27:]
                else:
                    sh_coeff = diffusion_face[..., -27:]
            # need to resize to 256x256
            source_image_256 =  torch.nn.functional.interpolate(
                batch['source_image'],
                size=(256,256),
                mode='bilinear', 
                align_corners=False
            )
            vae_features = self.get_latents_from_images(source_image_256)
            batch_size = vae_features.shape[0]
            vae_features = vae_features.view(batch_size, -1)

            # concat shcoeff if need 
            if self.feature_type in ['vae_shcoeff']:
                features = torch.cat([vae_features, sh_coeff], dim=-1)
            else:
                features = vae_features

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
        light_features = self.get_light_features(batch)
        set_light_direction(self.pipe.unet, light_features, is_apply_cfg=False) #B,C
        
        # this is intentional. we set didn't concat light into text prompt anymore
        encoder_hidden_states = prompt_embeds

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
        
        if self.use_false_shading:
            # now we will feed false shading to model 
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=self.get_control_image(batch, query_false_shading=True),
                conditioning_scale=self.condition_scale,
                guess_mode=False,
                return_dict=False,
            )
            negative_noise = self.pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            prev_timesteps = torch.clamp(timesteps - 1, 0, self.pipe.scheduler.config.num_train_timesteps-1)

            # predict latent with less noise 
            positive_latents = []
            negative_latents = []
            for batch_id in range(batch_size):
                positive_latents.append(self.pipe.scheduler.add_noise(latents[batch_id], noise[batch_id], timesteps[batch_id]))
                negative_latents.append(self.pipe.scheduler.add_noise(latents[batch_id], negative_noise[batch_id], timesteps[batch_id]))
            positive_latents = torch.stack(positive_latents)
            negative_latents = torch.stack(negative_latents)
            anchor_latents = self.pipe.scheduler.add_noise(latents, noise, prev_timesteps) # timestep before

            # compute triplet loss 
            triplet_loss = torch.nn.functional.triplet_margin_loss(
                anchor_latents, 
                positive_latents, 
                negative_latents, 
                margin=0.1, # torch suggest 1.0, but ChatGPT suggest 
                p=2, 
                reduction='mean'
            )
            loss += triplet_loss
        
        if self.use_triplet_background:
            positive_latents = []
            for batch_id in range(batch_size):
                positive_latents.append(self.pipe.scheduler.add_noise(latents[batch_id], noise[batch_id], timesteps[batch_id]))
            positive_latents = torch.stack(positive_latents)
            prev_timesteps = torch.clamp(timesteps - 1, 0, self.pipe.scheduler.config.num_train_timesteps-1)
            negative_latents = self.pipe.scheduler.add_noise(self.get_latents_from_images(batch['background']), noise, prev_timesteps) 
            anchor_latents = self.pipe.scheduler.add_noise(latents, noise, prev_timesteps) 
            # compute triplet loss 
            triplet_loss = torch.nn.functional.triplet_margin_loss(
                anchor_latents, 
                positive_latents, 
                negative_latents, 
                margin=0.1, # torch suggest 1.0, but ChatGPT suggest 
                p=2, 
                reduction='mean'
            )
            loss += triplet_loss



        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def get_control_image(self, batch, array_index=None, query_false_shading=False):
        background = batch['background']
        if not query_false_shading:
            shading = batch['shading']
        else:
            shading = batch['false_shading']
        if array_index is not None:
            background = background[array_index]
            shading = shading[array_index]
        return torch.cat([background, shading],dim=1)
    

            
    def select_batch_keyword(self, batch, keyword):            
        if self.feature_type in ["diffusion_face", "diffusion_face_shcoeff", 'clip_shcoeff', 'clip', "shcoeff_order2", "vae", "vae_shcoeff"]:
            batch['diffusion_face'] = batch[f'{keyword}_diffusion_face']
            batch['background'] = batch[f'{keyword}_background']
            batch['shading'] = batch[f'{keyword}_shading']
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        return batch
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):   
        try:
            log_dir = self.logger.log_dir
        except:
            log_dir = self.log_dir

        os.chmod(log_dir, 0o777)
        
        USE_LIGHT_DIRECTION_CONDITION = True

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
        #set_light_direction(self.pipe.unet, None,  is_apply_cfg=False)
        set_light_direction(self.pipe.unet, source_light_features,  is_apply_cfg=False)


        # Inversion to get z_noise 
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt = batch['text'],
            device = device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=is_apply_cfg,
            negative_prompt = [""] * len(batch['text'])
        )
        
        interrupt_index = int(self.num_inversion_steps * self.ddim_strength) if self.ddim_strength > 0 else None
        # get DDIM inversion  
        ddim_latents, ddim_timesteps = get_ddim_latents(
                pipe=self.pipe,
                image=batch['source_image'],
                text_embbeding=prompt_embeds,
                num_inference_steps=self.num_inversion_steps,
                generator=torch.Generator().manual_seed(self.seed),
                controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                guidance_scale=self.ddim_guidance_scale,
                interrupt_index = interrupt_index
            )

        # if dataset is not list, convert to list
        for key in ['target_ldr_envmap', 'target_norm_envmap', 'target_image', 'target_sh_coeffs', 'word_name']:
            if key in batch and not isinstance(batch[key], list):
                batch[key] = [batch[key]]

        if USE_LIGHT_DIRECTION_CONDITION:
            #Apply the target light direction
            self.select_batch_keyword(batch, 'target')    

        # compute inpaint from nozie
        mse_output = []
        noise = self.get_noise_from_latents(ddim_latents[-1])
        for target_idx in range(len(batch['target_image'])):
            target_light_features = self.get_light_features(batch, array_index=target_idx, generator=torch.Generator().manual_seed(self.seed))            
            # if target_idx > 0:
            set_light_direction(
                self.pipe.unet,
                target_light_features,
                is_apply_cfg=is_apply_cfg
            )

            pipe_args = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "output_type": "pt",
                "guidance_scale": self.guidance_scale,
                "return_dict": False,
                "num_inference_steps": self.num_inversion_steps,
                "generator": torch.Generator().manual_seed(self.seed),
              
            }
            if self.ddim_strength > 0:
                pipe_args["image"] = ddim_latents[interrupt_index]
                pipe_args["strength"] = self.ddim_strength                        
                ddim_pipe = self.pipe_img2img
                pipe_args["image"] = ddim_latents[interrupt_index]
                if hasattr(self.pipe, "controlnet"):
                    pipe_args["control_image"] = self.get_control_image(batch, array_index=target_idx) 
            else:
                pipe_args["latents"] = ((1.0 - self.gaussain_strength) * ddim_latents[-1]) + (self.gaussain_strength * noise)
                ddim_pipe = self.pipe
                if hasattr(self.pipe, "controlnet"):
                    pipe_args["image"] = self.get_control_image(batch, array_index=target_idx)  

            pt_image, _ = ddim_pipe(**pipe_args)
            
            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0 #bump back to range[0-1]

            gt_image = gt_image.to(pt_image.device)
            tb_image = [gt_image, pt_image]
            
            # generation only to see if everything still work as expected
            sd_args = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": self.guidance_scale,
                "output_type": "pt",
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed),
            }
            if hasattr(self.pipe, "controlnet"):
                sd_args["image"] = self.get_control_image(batch, array_index=target_idx)  
            # we store previous scheduler (likely DDIM) and switch to scheduler for regular generation (likely UniPC)
            prev_scheduler = self.pipe.scheduler 
            self.pipe.scheduler = self.regular_scheduler
            sd_image, _ = self.pipe(**sd_args)
            # reverse back to previous scheduler after generation
            self.pipe.scheduler = prev_scheduler
            tb_image.append(sd_image)

            # add control image to tensorbaord to see how good image is produce 
            if "image" in pipe_args:
                if pipe_args["image"].shape[1] % 3 == 0:
                    for i in range(pipe_args["image"].shape[1] // 3):
                        tb_image.append((pipe_args["image"][:, (i)*3:(i+1)*3]+1.0) / 2.0)
            
            images = torch.cat(tb_image, dim=0)
            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
            
            tb_name = f"{batch['name'][0].replace('/','-')}/{batch['word_name'][target_idx][0].replace('/','-')}"
            
            # calcuarte psnr 
            mse = torch.nn.functional.mse_loss(gt_image, pt_image, reduction="none").mean()
            mse_output.append(mse[None])
            psnr = -10 * torch.log10(mse)

            ssim = self._ssim_loss(gt_image, pt_image)
            ddsim = (1.0 - ssim) / 2.0

            lpips = None 
            if self._lpips_loss is not None:
                # lpips need to normalize to [-1,1]
                normalize_pt_image = pt_image * 2.0 - 1.0
                normalize_gt_image = gt_image * 2.0 - 1.0
                lpips = self._lpips_loss(normalize_pt_image, normalize_gt_image)
                self.log('lpips', lpips)
            
            self.logger.experiment.add_image(f'{tb_name}', image, self.global_step)
            self.log('psnr', psnr)
            self.log('ssim', ssim)
            self.log('ddsim', ddsim)
            self.log('mse', mse)


            if is_save_image:
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"

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
                # save psnr to file
                os.makedirs(f"{log_dir}/{epoch_text}psnr", exist_ok=True)
                with open(f"{log_dir}/{epoch_text}psnr/{filename}.txt", "w") as f:
                    f.write(f"{psnr.item()}\n")
                os.chmod(f"{log_dir}/{epoch_text}psnr", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}psnr/{filename}.txt", 0o777)

                # save prompt
                os.makedirs(f"{log_dir}/{epoch_text}prompt", exist_ok=True) 
                with open(f"{log_dir}/{epoch_text}prompt/{filename}.txt", 'w') as f:
                    f.write(batch['text'][0])
                os.chmod(f"{log_dir}/{epoch_text}prompt", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}prompt/{filename}.txt", 0o777)

                # save the source_image
                os.makedirs(f"{log_dir}/{epoch_text}target_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{log_dir}/{epoch_text}target_image/{filename}.jpg")
                os.chmod(f"{log_dir}/{epoch_text}target_image", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}target_image/{filename}.jpg", 0o777)
                if 'source_image' in batch:
                    os.makedirs(f"{log_dir}/{epoch_text}source_image", exist_ok=True)
                    source_image = (batch['source_image'][0] + 1.0) / 2.0 #bump back to range[0-1]
                    torchvision.utils.save_image(source_image, f"{log_dir}/{epoch_text}source_image/{filename}.jpg")
                    os.chmod(f"{log_dir}/{epoch_text}source_image", 0o777)
                    os.chmod(f"{log_dir}/{epoch_text}source_image/{filename}.jpg", 0o777)
                # save all score calcurateion
                os.makedirs(f"{log_dir}/{epoch_text}mse", exist_ok=True)
                with open(f"{log_dir}/{epoch_text}mse/{filename}.txt", "w") as f:
                    f.write(f"{mse.item()}\n")
                os.chmod(f"{log_dir}/{epoch_text}mse", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}mse/{filename}.txt", 0o777)
                os.makedirs(f"{log_dir}/{epoch_text}ssim", exist_ok=True)
                with open(f"{log_dir}/{epoch_text}ssim/{filename}.txt", "w") as f:
                    f.write(f"{ssim.item()}\n")
                os.chmod(f"{log_dir}/{epoch_text}ssim", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}ssim/{filename}.txt", 0o777)
                os.makedirs(f"{log_dir}/{epoch_text}ddsim", exist_ok=True)
                with open(f"{log_dir}/{epoch_text}ddsim/{filename}.txt", "w") as f:
                    f.write(f"{ddsim.item()}\n")
                os.chmod(f"{log_dir}/{epoch_text}ddsim", 0o777)
                os.chmod(f"{log_dir}/{epoch_text}ddsim/{filename}.txt", 0o777)
                if self._lpips_loss is not None:
                    os.makedirs(f"{log_dir}/{epoch_text}lpips", exist_ok=True)
                    with open(f"{log_dir}/{epoch_text}lpips/{filename}.txt", "w") as f:
                        f.write(f"{ddsim.item()}\n")
                    os.chmod(f"{log_dir}/{epoch_text}lpips", 0o777)
                    os.chmod(f"{log_dir}/{epoch_text}lpips/{filename}.txt", 0o777)

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
            {'params': self.adaptive_group_norm, 'lr': self.learning_rate},
            {'params': self.controlnet_trainable, 'lr': self.learning_rate * self.ctrlnet_lr},
        ])
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale

    def set_ddim_strength(self, ddim_strength):
        if not self.already_load_img2img:
            self.setup_ddim_img2img()
        self.ddim_strength = ddim_strength
        
    def set_gaussain_strength(self, gaussain_strength):
        self.gaussain_strength = gaussain_strength

class ScrathSDDiffusionFace(SDDiffusionFace):
    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # load main UNet
        unet = UNet2DConditionModel.from_config(sd_path, subfolder='unet')

        # create controlnet from unet. Unet take 6 channel input with are 3 channel of background and other 3 channel of shading
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6) 

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            unet=unet, # We add unet here to prevent it from reloading Unet
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()


class SDWithoutAdagnDiffusionFace(SDDiffusionFace):
    
    def get_light_features(self, *args, **kwargs):
        # We don't use light feature, we only use controlnet path
        return None
    
    def setup_light_block(self):
        self.pipe.to('cuda')


class SDOnlyAdagnDiffusionFace(SDDiffusionFace):
    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # load main UNet
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')

        # load pipeline
        self.pipe =  StableDiffusionPipeline.from_pretrained(
            sd_path,
            unet=unet, # We add unet here to prevent it from reloading Unet
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()
    
    def setup_trainable(self):
        # register trainable module to pytorch lighting model 

        # register adaptive group_norm
        adaptive_group_norm = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.adaptive_group_norm = torch.nn.ParameterList(adaptive_group_norm)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.adaptive_group_norm, 'lr': self.learning_rate},
        ])
        return optimizer

class SDDiffusionFaceNoBg(SDDiffusionFace):

    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # load main UNet
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')

        # Change control to 3channel  Unet take 6 channel input with are 3 channel of background and other 3 channel of shading
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3) 

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            unet=unet, # We add unet here to prevent it from reloading Unet
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

    def get_control_image(self, batch, array_index=None, query_false_shading=False):
        background = batch['background']
        if not query_false_shading:
            shading = batch['shading']
        else:
            shading = batch['false_shading']
        if array_index is not None:
            shading = shading[array_index]
        return shading
    
class SDDiffusionFaceNoShading(SDDiffusionFaceNoBg):

    def get_control_image(self, batch, array_index=None, query_false_shading=False):
        shading = batch['background']
        if array_index is not None:
            shading = shading[array_index]
        return shading
    
class SDOnlyShading(SDDiffusionFaceNoBg):
    
    def get_light_features(self, *args, **kwargs):
        # We don't use light feature, we only use controlnet path
        return None
    
    def setup_light_block(self):
        self.pipe.to('cuda')


class SDDiffusionFace5ch(SDDiffusionFace):
    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # load main UNet
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')

        # create controlnet from unet. Unet take 5 channel input with are 2 channel of background and other 3 channel of shading
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=5) 

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            unet=unet, # We add unet here to prevent it from reloading Unet
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()
