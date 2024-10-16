"""
AffineControl.py
Affine transform (Adaptive group norm) 
"""

import os 
import torch 
import torch.utils
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from tqdm.auto import tqdm
import ezexr
from PIL import Image
import bitsandbytes as bnb

from constants import *
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler, DDIMInverseScheduler, IFPipeline, IFImg2ImgPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from LightEmbedingBlock import set_light_direction, add_light_block

from InversionHelper import get_ddim_latents, get_text_embeddings, get_null_embeddings, apply_null_embedding
 
MASTER_TYPE = torch.float16

class StableDiffusionImg2ImgNoPrepLatentsPipeline(StableDiffusionImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        return image
    
class StableDiffusionControlNetNoPrepLatentsPipeline(StableDiffusionControlNetImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        return image
 
class AffineControl(L.LightningModule):

    def __init__(
            self,
            learning_rate=1e-4,
            guidance_scale=3.0,
            gate_multipiler=1,
            feature_type="vae",
            num_inversion_steps=500,
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
        #self.num_inversion_steps = num_inversion_steps
        # null-text inversion paper Section 4 ablation study Page 6 left side suggest to use 500 iteration with N=10
        self.use_null_text = False
        self.ddim_guidance_scale = 1.0
        self.ddim_strength = 0.0
        self.num_inversion_steps = 500
        self.num_null_text_steps = 10
        self.num_inference_steps = num_inference_steps
        self.save_hyperparameters()
        self.light_feature_indexs = [True] * self.num_inversion_steps

        self.already_load_img2img = False
        

        self.seed = 42
        self.is_plot_train_loss = True
        self.setup_sd()
        self.setup_ddim()
        self.setup_light_block()
        self.log_dir = ""

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


    def setup_light_block(self):
        if self.feature_type == "shcoeff_order2":
            mlp_in_channel = 27
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
        #  add light block to unet, 1024 is the shape of output of both LDR and HDR_Normalized clip combine
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')

        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())

        self.unet_trainable = torch.nn.ParameterList(unet_trainable)

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

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()

        #enable callback for NULL_TEXT INVERSION
        self.pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "latent_model_input", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "cond_scale", "guess_mode", "image"]


    def setup_ddim(self):
        self.normal_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
        self.pipe.scheduler = self.normal_scheduler
        
    
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

        if hasattr(self.pipe, "vae"):
            latents = self.pipe.vae.encode(batch['source_image']).latent_dist.sample().detach()
            latents = latents * self.pipe.vae.config.scaling_factor
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

            if isinstance(self.pipe, IFPipeline) or isinstance(self.pipe, IFImg2ImgPipeline):
                # THIS TRAINING won't support original schudler, please use DDIM from now on.
                model_pred, _ = model_pred.split(noisy_latents.shape[1], dim=1)

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

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
        elif self.feature_type == "shcoeff_order2":
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

        # precompute-variable
        is_apply_cfg = self.guidance_scale > 1
        epoch_text = f"epoch_{self.current_epoch:04d}/" if is_seperate_dir_with_epoch else ""
        source_name = f"{batch['name'][0].replace('/','-')}"
        
        # we explore problem that feature differnet here 
        print("CHECK IF LDR CLOSE TOGETHER?: ", torch.isclose(batch['source_ldr_envmap'], batch['target_ldr_envmap'][0]).all())
        print("CHECK IF NORM_HDR CLOSE TOGETHER?: ", torch.isclose(batch['source_norm_envmap'], batch['target_norm_envmap'][0]).all())
        print("CHECK IF LDR CLOSE TOGETHER?: ", torch.isclose(batch['source_ldr_envmap'], batch['target_ldr_envmap'][0]).all())
        # save image to see why it differnet
        print("SOURCE_ENVMAP: ", torch.max(batch['source_norm_envmap']), torch.min(batch['source_norm_envmap']))
        print("TARGET_ENVMAP: ", torch.max(batch['target_norm_envmap'][0]), torch.min(batch['target_norm_envmap'][0])) 
        torchvision.utils.save_image(batch['source_norm_envmap'], f"{log_dir}/source_norm_envmap.jpg")
        torchvision.utils.save_image(batch['target_norm_envmap'][0], f"{log_dir}/target_norm_envmap.jpg")
        self.select_batch_keyword(batch, 'source')
        source_light_features = self.get_light_features(batch, generator=torch.Generator().manual_seed(self.seed))
        self.select_batch_keyword(batch, 'target')
        target_light_features = self.get_light_features(batch, array_index=0, generator=torch.Generator().manual_seed(self.seed))
        print("CHECK IF FEATURE CLOSE TOGETHER?: ", torch.isclose(target_light_features, source_light_features).all())
        exit()

        # Apply the source light direction
        self.select_batch_keyword(batch, 'source')
        # save image from ['source_ldr_envmap']
        if USE_LIGHT_DIRECTION_CONDITION:
            source_light_features = self.get_light_features(batch, generator=torch.Generator().manual_seed(self.seed))
            set_light_direction(
                self.pipe.unet, 
                source_light_features, 
                is_apply_cfg=False #during DDIM inversion, we don't want to apply the cfg
            )
        else:
            source_light_features = None
            set_light_direction(
            self.pipe.unet, 
                None, 
                is_apply_cfg=False
            )

        # compute text embedding
        text_embbeding = get_text_embeddings(self.pipe, batch['text']).to(self.pipe.unet.dtype)
        negative_embedding = get_text_embeddings(self.pipe, '').to(self.pipe.unet.dtype)
        
        if os.path.exists(f"{log_dir}/ddim_latents/{source_name}.pt"):
            # skip the ddim_latents if it already exists
            ddim_latents = torch.load(f"{log_dir}/ddim_latents/{source_name}.pt")
        else:
            interrupt_index = int(self.num_inversion_steps * self.ddim_strength) if self.ddim_strength > 0 else None
            if interrupt_index is not None:
                interrupt_index = np.clip(interrupt_index, 0, 999)
            # get DDIM inversion
            ddim_latents, ddim_timesteps = get_ddim_latents(
                pipe=self.pipe,
                image=batch['source_image'],
                text_embbeding=text_embbeding,
                num_inference_steps=self.num_inversion_steps,
                generator=torch.Generator().manual_seed(self.seed),
                controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                guidance_scale=self.ddim_guidance_scale,
                interrupt_index=interrupt_index
            )
            if is_save_image:
                # save ddim_latents to file
                os.makedirs(f"{log_dir}/ddim_latents", exist_ok=True)
                torch.save(ddim_latents, f"{log_dir}/ddim_latents/{source_name}.pt.pt")
                torch.save(ddim_timesteps, f"{log_dir}/ddim_latents/{source_name}_timesteps.pt")

        if self.use_null_text and self.guidance_scale > 1:                
            if os.path.exists(f"{log_dir}/null_embeddings/{source_name}.pt"):
                # skip the null embeddings if it already exists
                null_embeddings = torch.load(f"{log_dir}/{epoch_text}null_embeddings/{source_name}.pt")
                null_latents = torch.load(f"{log_dir}/{epoch_text}null_latents/{source_name}.pt")
            else:
                set_light_direction(
                    self.pipe.unet,
                    source_light_features,
                    is_apply_cfg=True # Disable cfg for fast inversion
                )
                
                def before_positive_pass_callback(): 
                    set_light_direction(
                        self.pipe.unet,
                        source_light_features,
                        is_apply_cfg=False # Disable cfg for fast inversion
                    )

                def before_negative_pass_callback():
                    set_light_direction(
                        self.pipe.unet,
                        None,
                        is_apply_cfg=False # Disable cfg for fast inversion
                    )

                def before_final_denoise_callback():
                    set_light_direction(
                        self.pipe.unet,
                        source_light_features,
                        is_apply_cfg=True # Disable cfg for fast inversion
                    ) 
                
                def after_final_denoise_callback():
                    set_light_direction(
                        self.pipe.unet,
                        source_light_features,
                        is_apply_cfg=True # Disable cfg for fast inversion
                    )

                null_embeddings, null_latents = get_null_embeddings(
                    self.pipe,
                    ddim_latents=ddim_latents,
                    text_embbeding=text_embbeding,
                    negative_embedding=negative_embedding,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inversion_steps,
                    controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                    num_null_optimization_steps=self.num_null_text_steps,
                    generator=torch.Generator().manual_seed(self.seed),
                    before_positive_pass_callback = before_positive_pass_callback,
                    before_negative_pass_callback = before_negative_pass_callback,
                    before_final_denoise_callback = before_final_denoise_callback,
                    after_final_denoise_callback = after_final_denoise_callback,
                )
            if is_save_image:
                # save image from null_latents 
                null_latent = null_latents[-1]
                if hasattr(self.pipe, "vae"):
                    image = self.pipe.vae.decode(null_latent / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                else:
                    image = self.pipe.unet(null_latent, return_dict=False)[0]
                # rescale image to [0,1]
                image = (image / 2 + 0.5).clamp(0, 1)
                # save image to tensorboard
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][0][0].replace('/','-')}"
                os.makedirs(f"{log_dir}/{epoch_text}null_latents", exist_ok=True)
                torchvision.utils.save_image(image, f"{log_dir}/{epoch_text}null_latents/{filename}.jpg")

                # save null_latents to file
                os.makedirs(f"{log_dir}/{epoch_text}null_latents", exist_ok=True)
                torch.save(null_latents, f"{log_dir}/{epoch_text}null_latents/{source_name}.pt")

                # save null_embedigns to file
                os.makedirs(f"{log_dir}/{epoch_text}null_embeddings", exist_ok=True)
                torch.save(null_embeddings, f"{log_dir}/{epoch_text}null_embeddings/{source_name}.pt")

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
            target_light_features = self.get_light_features(batch, array_index=target_idx, generator=torch.Generator().manual_seed(self.seed))            
            if target_idx > 0:                
                set_light_direction(
                    self.pipe.unet,
                    target_light_features,
                    is_apply_cfg=is_apply_cfg
                )
            
            if self.use_null_text and self.guidance_scale > 1:
                pt_image = apply_null_embedding(
                    self.pipe,
                    ddim_latents[-1],
                    null_embeddings,
                    text_embbeding,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inversion_steps,
                    generator=torch.Generator().manual_seed(self.seed),
                    controlnet_image=self.get_control_image(batch) if hasattr(self.pipe, "controlnet") else None,
                    null_latents=null_latents
                )
            else:
                pipe_args = {
                    "prompt_embeds": text_embbeding,
                    "negative_prompt_embeds": negative_embedding,
                    "output_type": "pt",
                    "guidance_scale": self.guidance_scale,
                    "return_dict": False,
                    "num_inference_steps": self.num_inversion_steps,
                    "generator": torch.Generator().manual_seed(self.seed)
                }
                if isinstance(self.pipe, IFImg2ImgPipeline): # support for deepfloyd
                    pipe_args["image"] = ddim_latents[-1]
                    pipe_args["strength"] = 1.0
                    ddim_pipe = self.pipe
                    self.pipe.scheduler.config.variance_type = "place_holder"
                    pt_image, _, _ = ddim_pipe(**pipe_args)
                else: #support other pipeline
                    if self.ddim_strength > 0: # support denoise but not all the way
                        pipe_args["strength"] = self.ddim_strength                        
                        ddim_pipe = self.pipe_img2img
                        pipe_args["image"] = ddim_latents[interrupt_index]
                        if hasattr(self.pipe, "controlnet"):
                            pipe_args["control_image"] = self.get_control_image(batch)  
                    else: # original DDIM
                        pipe_args["latents"] = ddim_latents[-1]
                        ddim_pipe = self.pipe
                        if hasattr(self.pipe, "controlnet"):
                            pipe_args["image"] = self.get_control_image(batch)    
                        # support callback to swap source and light
                        timesteps = []
                        step_ids = []
                        def callback_swap_source_light(pipe, step_index, timestep, callback_kwargs):
                            timesteps.append(timestep)
                            step_ids.append(step_index)
                            swap_id = self.get_swap_light_id(step_index, timestep)
                            features = source_light_features if swap_id == "SOURCE" else target_light_features
                            set_light_direction(
                                pipe.unet,
                                features,
                                is_apply_cfg=is_apply_cfg
                            )
                            return callback_kwargs
                        pipe_args["callback_on_step_end"] = callback_swap_source_light

                    pt_image, _ = ddim_pipe(**pipe_args)
                    

            gt_on_batch = 'target_image' if "target_image" in batch else 'source_image'
            gt_image = (batch[gt_on_batch][target_idx] + 1.0) / 2.0

            #deepfloyd need to resize the image
            if hasattr(self, "pipe_upscale"):
                # save thing for upscale 
                if is_save_image:
                    os.makedirs(f"{log_dir}/{epoch_text}upscale", exist_ok=True)
                    # save pt_image, text_embedding and negative_embedding as torch
                    filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                    torch.save(pt_image, f"{log_dir}/{epoch_text}upscale/{filename}_pt_image.pt")
                    torch.save(text_embbeding, f"{log_dir}/{epoch_text}upscale/{filename}_text_embbeding.pt")
                    torch.save(negative_embedding, f"{log_dir}/{epoch_text}upscale/{filename}_negative_embedding.pt")
                    
                if False:
                    pt_image = self.pipe_upscale(
                        image=pt_image,
                        prompt_embeds=text_embbeding,
                        negative_prompt_embeds=negative_embedding,
                        output_type="pt"
                    ).images
                    pt_image = torch.nn.functional.interpolate(pt_image, size=gt_image.shape[-2:], mode="bilinear", align_corners=False) 
                else:
                    pt_image = (pt_image + 1.0) / 2.0
                    pt_image = torch.clamp(pt_image, 0, 1)
                    gt_image = torch.nn.functional.interpolate(gt_image, size=pt_image.shape[-2:], mode="bilinear", align_corners=False)

            if isinstance(self.pipe, IFPipeline) or isinstance(self.pipe, IFImg2ImgPipeline):
                if is_save_image:
                    os.makedirs(f"{log_dir}/{epoch_text}upscale", exist_ok=True)
                    # save pt_image, text_embedding and negative_embedding as torch
                    filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                    torch.save(pt_image, f"{log_dir}/{epoch_text}upscale/{filename}_pt_image.pt")
                    torch.save(text_embbeding, f"{log_dir}/{epoch_text}upscale/{filename}_text_embbeding.pt")
                    torch.save(negative_embedding, f"{log_dir}/{epoch_text}upscale/{filename}_negative_embedding.pt")               
                pt_image = (pt_image + 1.0) / 2.0
                pt_image = torch.clamp(pt_image, 0, 1)
                gt_image = torch.nn.functional.interpolate(gt_image, size=pt_image.shape[-2:], mode="bilinear", align_corners=False)
              

            gt_image = gt_image.to(pt_image.device)
            tb_image = [gt_image, pt_image]

            if hasattr(self.pipe, "controlnet"):
                ctrl_image = self.get_control_image(batch)
                if isinstance(ctrl_image, list):
                    tb_image += ctrl_image
                else:
                    tb_image.append(ctrl_image)
            
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
                # save controlnet 
                if hasattr(self.pipe, "controlnet"):
                    os.makedirs(f"{log_dir}/{epoch_text}control_image", exist_ok=True)
                    if isinstance(ctrl_image, list):
                        for i, c in enumerate(ctrl_image):
                            torchvision.utils.save_image(c, f"{log_dir}/{epoch_text}control_image/{filename}_{i}.jpg")
                    else:
                        torchvision.utils.save_image(ctrl_image, f"{log_dir}/{epoch_text}control_image/{filename}.jpg")
                # save the target_ldr_envmap
                if hasattr(batch, "target_ldr_envmap"):
                    os.makedirs(f"{log_dir}/{epoch_text}target_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_ldr_envmap'][target_idx], f"{log_dir}/{epoch_text}target_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "target_norm_envmap"):
                    # save the target_norm_envmap
                    os.makedirs(f"{log_dir}/{epoch_text}target_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['target_norm_envmap'][target_idx], f"{log_dir}/{epoch_text}target_norm_envmap/{filename}.jpg")
                if hasattr(batch, "source_ldr_envmap"):
                    # save the source_ldr_envmap
                    os.makedirs(f"{log_dir}/{epoch_text}source_ldr_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_ldr_envmap'], f"{log_dir}/{epoch_text}source_ldr_envmap/{filename}.jpg")
                if hasattr(batch, "source_norm_envmap"):
                    # save the source_norm_envmap
                    os.makedirs(f"{log_dir}/{epoch_text}source_norm_envmap", exist_ok=True)
                    torchvision.utils.save_image(batch['source_norm_envmap'], f"{log_dir}/{epoch_text}source_norm_envmap/{filename}.jpg")
                
                # save prompt
                os.makedirs(f"{log_dir}/{epoch_text}prompt", exist_ok=True) 
                with open(f"{log_dir}/{epoch_text}prompt/{filename}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{log_dir}/{epoch_text}source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{log_dir}/{epoch_text}source_image/{filename}.png")
            if True:
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
        ])
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale
    
    def set_ddim_guidance(self, guidance_scale):
        self.ddim_guidance_scale = guidance_scale

    def set_inversion_step(self, num_inversion_steps):
        self.num_inversion_steps = num_inversion_steps
    
    def disable_null_text(self):
        self.use_null_text = False

    def set_ddim_strength(self, ddim_strength):
        if not self.already_load_img2img:
            self.setup_ddim_img2img()
        self.ddim_strength = ddim_strength

    def get_swap_light_id(self, step_index, timestep):
        return "TARGET" if self.light_feature_indexs[step_index] else "SOURCE"
    
    def set_light_feature_indexs(self, light_feature_indexs):
        self.light_feature_indexs = light_feature_indexs