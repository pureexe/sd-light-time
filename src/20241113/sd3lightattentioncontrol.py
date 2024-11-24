"""
sd3lightattentioncontrol.py
Stable Diffusion 3 with light control similar to prompt control
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
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import pt_to_pil
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from light_injector.transformer import inject_transformer

from sd3RFinversion import interpolated_inversion, interpolated_denoise, encode_imgs, decode_imgs
from transformers import T5EncoderModel, BitsAndBytesConfig

MASTER_TYPE = torch.float16

class SD3LightAttentionControl(L.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            guidance_scale=3.5,
            feature_type="shcoeff_order2",
            num_inversion_steps=28,
            num_inference_steps=28,
            rf_gamma = 0.5,
            rf_eta_base = 0.95,
            rf_eta_trend = 'constant',
            rf_start_step = 0,
            rf_end_step = 9,
            *args,
            **kwargs
        ) -> None:
        super().__init__()
        self.condition_scale = 1.0
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.feature_type = feature_type

        # rf related config
        self.rf_gamma = rf_gamma
        self.rf_eta_base = rf_eta_base
        self.rf_eta_trend = rf_eta_trend
        self.rf_start_step = rf_start_step
        self.rf_end_step = rf_end_step

        self.num_inversion_steps = num_inversion_steps
        self.num_inference_steps = num_inference_steps
        self.save_hyperparameters()

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
        self.setup_light_block()
        self.log_dir = ""


    def setup_light_block(self):
        if self.feature_type == "shcoeff_order2":
            mlp_in_channel = 27
        elif self.feature_type == "shcoeff_fuse":
            mlp_in_channel = 643
        elif self.feature_type == "vae":
            mlp_in_channel = 32*32*4*2
        elif self.feature_type == "vae128":
            mlp_in_channel = 16*16*4*2
        elif self.feature_type == "vae64":
            mlp_in_channel = 8*8*4*2
        elif self.feature_type == "vae32":
            mlp_in_channel = 4*4*4*2
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        inject_transformer(self.pipe.transformer, in_channel=mlp_in_channel)
        self.pipe.to('cuda')

        # filter trainable layer
        transfomer_trainable = filter(lambda p: p.requires_grad, self.pipe.transformer.parameters())

        self.transfomer_trainable = torch.nn.ParameterList(transfomer_trainable)

    def setup_sd(self, sd_path="stabilityai/stable-diffusion-3.5-medium", controlnet_path=None):
        if controlnet_path is not None:
            raise Exception("ControlNet currently not support ")
        
        # recommend: drop T5 encoder to save memory as mention in https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#dropping-the-t5-text-encoder-during-inference


        DISABLE_T5 = True
        T5_8BIT = False


        pipe_args = {
            'pretrained_model_name_or_path': sd_path,
            'safety_checker': None, 
            'torch_dtype': MASTER_TYPE
        }

        if DISABLE_T5:
            pipe_args['text_encoder_3'] = None 
            pipe_args['tokenizer_3'] = None
        elif T5_8BIT:
            #load T5 at 8bit 
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_3 = T5EncoderModel.from_pretrained(
                sd_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            pipe_args['text_encoder_3'] = text_encoder_3 
            


        # load pipeline
        self.pipe =  StableDiffusion3Pipeline.from_pretrained(**pipe_args)

        # load transformer from pretrain 
        self.pipe.transformer.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)
        if self.pipe.text_encoder_3 is not None: 
            self.pipe.text_encoder_3.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

        self.noise_scheduler_copy = copy.deepcopy(self.pipe.scheduler)



        
    
    def set_seed(self, seed):
        self.seed = seed
   
    def get_vae_features(self, images, generator=None):
        assert images.shape[1] == 3, "Only support RGB image"
        #assert images.shape[2] == 256 and images.shape[3] == 256, "Only support 256x256 image"
        with torch.inference_mode():
            # VAE need input in range of [-1,1]
            assert images.min() >= 0 and images.max() <= 1, "Image should be in range of [0,1]"
            images = images * 2.0 - 1.0
            vae = self.pipe.vae if hasattr(self.pipe, "vae") else self.vae
            emb = vae.encode(images).latent_dist.sample(generator=generator) * vae.config.scaling_factor 
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb
        
        
    def get_light_features(self, batch, array_index=None, generator=None):
        if self.feature_type in ["vae", "vae128", "vae64", "vae32"]:
            ldr_envmap = batch['ldr_envmap']
            norm_envmap = batch['norm_envmap']
            if array_index is not None:
                ldr_envmap = ldr_envmap[array_index]
                norm_envmap = norm_envmap[array_index]

            if len(self.feature_type) > 3 and self.feature_type[3:].isdigit():
                image_size = int(self.feature_type[3:])
                # resize image to size 
                ldr_envmap = torch.nn.functional.interpolate(ldr_envmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
                norm_envmap = torch.nn.functional.interpolate(norm_envmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
            ldr_features = self.get_vae_features(ldr_envmap, generator)
            hdr_features = self.get_vae_features(norm_envmap, generator)
            # concat ldr and hdr features
            return torch.cat([ldr_features, hdr_features], dim=-1)    
        elif self.feature_type in ["shcoeff_order2","shcoeff_fuse"]:
            shcoeff = batch['sh_coeffs']
            if array_index is not None:
                shcoeff = shcoeff[array_index]
            shcoeff = shcoeff.view(shcoeff.size(0), -1) #flatten shcoeff to [B, 27]
            return shcoeff[:,None] #attention mechanism need shape [B,TOKEN,D], we treat light as 1 token
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.pipe.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.pipe.device)
        timesteps = timesteps.to(self.pipe.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def compute_train_loss(self, batch, batch_idx, timesteps=None, seed=None):
        USE_PRECONDITION_OUTPUT = True 

        # get light features 
        light_features = self.get_light_features(batch)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.pipe.encode_prompt(
            prompt=batch['text'], 
            prompt_2=batch['text'],
            prompt_3=batch['text']
        )

        if hasattr(self.pipe, "vae"):
            latents = self.pipe.vae.encode(batch['source_image']).latent_dist.sample().detach()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        else:
            # resize source image to 64x64
            latents = torch.nn.functional.interpolate(batch['source_image'], size=(64, 64), mode="bilinear", align_corners=False)

        # Sample noise that we'll add to the latents
        if seed is not None:
            # create random genration seed
            torch.manual_seed(seed)
            noise = torch.randn_like(latents, memory_format=torch.contiguous_format)
        else:
            noise = torch.randn_like(latents)
        
        model_input = latents
        bsz = model_input.shape[0]


        # we have to do Flow matching loss here
        # Code taken from train_controlnet_sd3.py
        # @see https://github.com/huggingface/diffusers/blob/13e8fdecda91e27e40b15fa8a8f456ade773e6eb/examples/controlnet/train_controlnet_sd3.py#L1252

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )

        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps =  self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Predict the noise residual
        model_pred = self.pipe.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            light_hidden_states=light_features,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        if USE_PRECONDITION_OUTPUT:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)

        # flow matching loss
        if USE_PRECONDITION_OUTPUT:
            target = model_input
        else:
            target = noise - model_input

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def get_control_image(self, batch):
        raise NotImplementedError("get_control_image must be implemented")
    
    def select_batch_keyword(self, batch, keyword):
                    
        if self.feature_type.startswith("vae") :
            batch['ldr_envmap'] = batch[f'{keyword}_ldr_envmap']
            batch['norm_envmap'] = batch[f'{keyword}_norm_envmap']
        elif self.feature_type in ["shcoeff_order2", "shcoeff_fuse"]:
            batch['sh_coeffs'] = batch[f'{keyword}_sh_coeffs']
        else:
            raise ValueError(f"feature_type {self.feature_type} is not supported")
        return batch
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False, is_seperate_dir_with_epoch=False):           
        try:
            log_dir = self.logger.log_dir
        except:
            log_dir = self.log_dir

        USE_LIGHT_DIRECTION_CONDITION = True
        CHECK_IF_REGULAR_GENERATION_NO_COLLAPSE = True

        # precompute-variable
        is_apply_cfg = self.guidance_scale > 1
        epoch_text = f"step_{self.global_step:06d}/" if is_seperate_dir_with_epoch else ""
        source_name = f"{batch['name'][0].replace('/','-')}"

        # Apply the source light direction
        self.select_batch_keyword(batch, 'source')

        # set direction for inversion        
        source_light_features = self.get_light_features(batch, generator=torch.Generator().manual_seed(self.seed))
        source_light_features = source_light_features if USE_LIGHT_DIRECTION_CONDITION else None

        current_prompt = batch['text']

        # Inversion to get z_noise 
        source_latents = encode_imgs(batch['source_image'], self.pipe, self.pipe.dtype)
        inversed_latents = interpolated_inversion(
            self.pipe, 
            source_latents,
            self.rf_gamma,
            source_latents.dtype,
            prompt = current_prompt, 
            light_features = source_light_features,
            num_steps=self.num_inversion_steps,
            seed=self.seed
        )

        # if dataset is not list, convert to list
        for key in ['target_ldr_envmap', 'target_norm_envmap', 'target_image', 'target_sh_coeffs', 'word_name']:
            if key in batch and not isinstance(batch[key], list):
                batch[key] = [batch[key]]

        if USE_LIGHT_DIRECTION_CONDITION:
            #Apply the target light direction
            self.select_batch_keyword(batch, 'target')  

        # generate image without relighting to see if the model is not collaspe
        if CHECK_IF_REGULAR_GENERATION_NO_COLLAPSE:
            pt_latents =  interpolated_denoise(
                self.pipe, 
                source_latents,
                self.rf_eta_base,                    # base eta value
                'constant',                   # constant, linear_increase, linear_decrease
                0,                  # 0-based indexing, closed interval
                1,                    # 0-based indexing, open interval
                None,            # can be none if not using inversed latents
                use_inversed_latents=False,
                guidance_scale=self.guidance_scale,
                prompt=current_prompt,
                light_features = source_light_features,
                DTYPE=source_latents.dtype,
                num_steps=28,
                seed=self.seed
            )
            pt_image = decode_imgs(pt_latents, self.pipe) #[range 0-1] shape[b,c,h,w]
            image = pt_to_pil(pt_image)[0]
            filename = f"{batch['name'][0].replace('/','-')}"
            os.makedirs(f"{log_dir}/{epoch_text}from_prompt", exist_ok=True)
            torchvision.utils.save_image(pt_image, f"{log_dir}/{epoch_text}from_prompt/{filename}.jpg")


        # compute inpaint from nozie
        mse_output = []
        for target_idx in range(len(batch['target_image'])):
            target_light_features = self.get_light_features(batch, array_index=target_idx, generator=torch.Generator().manual_seed(self.seed))            
            relit_latents =  interpolated_denoise(
                self.pipe, 
                source_latents,
                self.rf_eta_base,                    # base eta value
                self.rf_eta_trend,                   # constant, linear_increase, linear_decrease
                self.rf_start_step,                  # 0-based indexing, closed interval
                self.rf_end_step,                    # 0-based indexing, open interval
                inversed_latents,            # can be none if not using inversed latents
                use_inversed_latents=True,
                guidance_scale=self.guidance_scale,
                prompt=current_prompt,
                light_features = source_light_features,
                DTYPE=source_latents.dtype,
                num_steps=28,
                seed=self.seed
            )
            # need to re-generate without inversion 

            pt_image = decode_imgs(relit_latents, self.pipe) #[range 0-1] shape[b,c,h,w]
            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0 #bump back to range[0-1]

            gt_image = gt_image.to(pt_image.device)
            tb_image = [gt_image, pt_image]
            
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
            {'params': self.transfomer_trainable, 'lr': self.learning_rate},
        ])
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale