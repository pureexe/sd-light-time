"""
AffineConsistancy.py
Affine transform (Adaptive group norm) that condition with environment map passthrough the VAE 
We also provide the consistancy loss from the chrome ball.
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

from constants import *
from diffusers import StableDiffusionPipeline, ControlNetModel

from  ball_helper import create_circle_tensor
from LightEmbedingBlock import set_light_direction, add_light_block, set_gate_shift_scale
from UnsplashLiteDataset import log_map_to_range
 
 
class AffineConsistancy26(L.LightningModule):
    def __init__(self, learning_rate=1e-3, gate_multipiler=1, guidance_scale=3.0, use_consistancy_loss=False, *args, **kwargs) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.learning_rate = learning_rate
        self.use_consistancy_loss = use_consistancy_loss
        self.save_hyperparameters()
        MASTER_TYPE = torch.float16

        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None, torch_dtype=MASTER_TYPE)

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        # load controlnet from pretrain
        if self.use_consistancy_loss:
            self.pipe.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
            self.pipe.controlnet.requires_grad_(False)
            self.pipe.controlnet.to('cuda')

        mlp_in_channel = 32*32*4*2

        #  add light block to unet, 1024 is the shape of output of both LDR and HDR_Normalized clip combine
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')


        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad and (len(p.shape) > 1 or p.shape[0] > 1), self.pipe.unet.parameters())
        gate_trainable = filter(lambda p: p.requires_grad and (len(p.shape) == 1 and p.shape[0] == 1), self.pipe.unet.parameters())

        self.unet_trainable = torch.nn.ParameterList(unet_trainable)
        self.gate_trainable = torch.nn.ParameterList(gate_trainable)

        #create circle mask 
        with torch.no_grad():
            self.circle_mask = create_circle_tensor(16, 16)

        self.seed = 42
        self.is_plot_train_loss = True
        self.gate_multipiler = gate_multipiler

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
            emb = self.pipe.vae.encode(images).latent_dist.sample(generator=generator) #* self.pipe.vae.config.scaling_factor 
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb

        
    def get_light_features(self, ldr_images, normalized_hdr_images, generator=None):
        ldr_features = self.get_vae_features(ldr_images, generator)
        hdr_features = self.get_vae_features(normalized_hdr_images, generator)
        # concat ldr and hdr features
        return torch.cat([ldr_features, hdr_features], dim=-1)

    def get_envmap_consistancy_loss(self, source_latent, target_image, noise, timesteps):
        """compute consistancy loss from chrome ball
        Given: target_image, 
        1. encode the target_image to target_latent with vae 
        2. add noise to the target_latent to be at the same timestep
        3. mask both source latent and target latent with circle mask
        4. compute mse (only inside the mask)

        Args:
            source_latent (torch.tensor): noise latent 
            target_image (torch.tensor): target image in range of [-1, 1]
            noise (torch.tensor): gaussian noise in channel of latent
            timesteps (torch.tensor): diffusion timesteps 

        Returns:
            torch.tensor: envmap consistancy loss
        """
        
        # check if the target_image is in range of [-1, 1]
        assert target_image.min() >= -1.0 and target_image.max() <= 1.0, "Only support [-1, 1] range image"    

        # check if target_image is 512x512         
        assert target_image.shape[2] == 512 and target_image.shape[3] == 512, "Only support 512x512 image"

        # check if source_latent is 64x64
        assert source_latent.shape[2] == 64 and source_latent.shape[3] == 64, "Only support 64x64 latent"


        # encode the target_image to target_latent with vae
        target_latent = self.pipe.vae.encode(target_image).latent_dist.sample().detach()
        target_latent = target_latent * self.pipe.vae.config.scaling_factor        
        noisy_latents = self.pipe.scheduler.add_noise(target_latent, noise, timesteps)

        # compute loss only the middle of latent size 16
        middle_source_latent = source_latent[:,:,24:40,24:40]
        middle_target_latent = noisy_latents[:,:,24:40,24:40]

        # apply circle mask on source and target latent
        mask = self.circle_mask.to(source_latent.device)
        middle_source_latent = middle_source_latent * mask
        middle_target_latent = middle_target_latent * mask

        # calculate consistancy loss
        loss = torch.nn.functional.mse_loss(middle_source_latent, middle_target_latent, reduction="mean")

        return loss
    
    def inpaint_chromeball(self, latents, depth_map, prompt_embeds, t, cond_scale=1.0):
        """
        forward controlnet for 1 time step to inpaint the chrome ball

        Args:
            latents (_type_): _description_
            depth_map (_type_): depth_map in range of [-1, 1] (B,3,512,512)
            prompt_embeds (_type_): text embedding
            t (_type_): diffusion timesteps
            cond_scale (float, optional): Controlnet scale. Defaults to 1.0.

        Returns:
            _type_: _description_
        """

        # forward the controlnet
        down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=depth_map,
            conditioning_scale=cond_scale,
            guess_mode=False,
            return_dict=False,
        )
        
        ctrl_pred = self.pipe.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        next_latents = self.pipe.scheduler.step(ctrl_pred, t, latents, return_dict=False)[0] # compute next latent (t-1 latent)

        return next_latents

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
            # Sample a random timestep for each image
            # sadly, schuduler.step does not support different timesteps for each image, so we use same time step for entire batch
            if self.use_consistancy_loss: 
                timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (1,), device=latents.device)
                timesteps = timesteps.expand(bsz)
                timesteps = timesteps.long().to(latents.device)
            else:
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
        light_features = self.get_light_features(batch['ldr_envmap'],batch['norm_envmap'])
        set_light_direction(self.pipe.unet, light_features, is_apply_cfg=False) #B,C

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        diffusion_loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
        if self.use_consistancy_loss:
            # inpainting chrome ball to the model_pred 
            self.pipe.scheduler.set_timesteps(1000) # set time step to 1000 for training step
            next_latents = self.pipe.scheduler.step(model_pred, timesteps[0], latents, return_dict=False)[0] # compute next latent (t-1 latent)
            chromeball_pred = self.inpaint_chromeball(next_latents, batch['control_depth'], encoder_hidden_states, torch.clamp(timesteps[0] - 1, min = 0))
            envconsistancy_loss = self.get_envmap_consistancy_loss(chromeball_pred, batch['chromeball_image'], noise, torch.clamp(timesteps[0] - 2, min = 0))

            loss = diffusion_loss + envconsistancy_loss
        else:
            loss = diffusion_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_train_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def generate_video_light(self):
        MAP_SIZE = 256
        # read global environment map 
        ## ldr
        ldr = torchvision.io.read_image("datasets/studio_small_05/studio_small_05.png") 
        ldr = ldr / 255.0 # normalize to [0,1]
        ldr = ldr[:3]
        ldr = torchvision.transforms.functional.resize(ldr, (MAP_SIZE,MAP_SIZE))
        
        #ldr = ldr.numpy().transpose(1, 2, 0)
        ## normalized_hdr
        hdr_normalized = ezexr.imread("datasets/studio_small_05/studio_small_05.exr")
        hdr_normalized = log_map_to_range(hdr_normalized)
        hdr_normalized = hdr_normalized.permute(2, 0, 1)
        hdr_normalized = hdr_normalized[:3] #only first 3 channel
        hdr_normalized = torchvision.transforms.functional.resize(hdr_normalized, (MAP_SIZE, MAP_SIZE))
        
        #hdr_normalized = hdr_normalized.numpy().transpose(1, 2, 0)


        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        # generate 8 face with 32 frame
        TOTAL_FACE = 8
        for face_id in range(TOTAL_FACE):
            print("GENERATING FACE ID: ", face_id)
            if self.use_set_guidance_scale:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/g{self.guidance_scale:.2f}/{face_id}"
            else:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{face_id}"
            os.makedirs(output_dir, exist_ok=True)
            VID_FRAME = 48
            VID_BATCH = 4
            output_frames = []

            need_cfg = self.guidance_scale > 1
            for vid_batch_id in range(VID_FRAME // VID_BATCH):
                rolled_hdr_normalized = []
                rolled_ldr = []
                for frame_id in range(VID_BATCH):
                    roll_ratio = (vid_batch_id * VID_BATCH + frame_id) / VID_FRAME
                    roll_cnt = int(roll_ratio * MAP_SIZE)
                    rolled_ldr.append(torch.roll(ldr, roll_cnt, dims=(-1))[None]) #C,H,W
                    rolled_hdr_normalized.append(torch.roll(hdr_normalized, roll_cnt, dims=(-1))[None]) #C,H,W
                rolled_hdr_normalized = torch.cat(rolled_hdr_normalized, dim=0).to('cuda').float()
                rolled_ldr = torch.cat(rolled_ldr, dim=0).to('cuda').float()
                light_features = self.get_light_features(rolled_ldr, rolled_hdr_normalized)
                set_light_direction(self.pipe.unet, light_features, is_apply_cfg=need_cfg)
                image, _ = self.pipe(
                    prompts[f"{face_id:05d}"],
                    num_images_per_prompt=VID_BATCH,
                    output_type="pil",
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=50,
                    return_dict = False,
                    generator=[torch.Generator().manual_seed(self.seed) for _ in range(VID_BATCH)]
                )
                for frame_id in range(VID_BATCH):
                    image[frame_id].save(f"{output_dir}/{(VID_BATCH*vid_batch_id) + frame_id:03d}.png")
                    output_frames.append(torchvision.transforms.functional.pil_to_tensor(image[frame_id])[None,None]) #B,T,C,H,W
            output_frames = torch.cat(output_frames, dim=1)
            self.logger.experiment.add_video(f'face/{face_id:03d}', output_frames, self.global_step, fps=6)
        
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        set_light_direction(self.pipe.unet, self.get_light_features(batch['ldr_envmap'],batch['norm_envmap']), is_apply_cfg=True)
        pt_image, _ = self.pipe(
            batch['text'], 
            output_type="pt",
            guidance_scale=self.guidance_scale,
            num_inference_steps=50,
            return_dict = False,
            generator=torch.Generator().manual_seed(self.seed)
        )
        gt_image = (batch["source_image"] + 1.0) / 2.0
        images = torch.cat([gt_image, pt_image], dim=0)
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
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
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text(f'text/{batch["word_name"][0]}', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
            self.logger.experiment.add_text('gate_multipiler', str(self.gate_multipiler), self.global_step)
            print("gate_multipiler", self.gate_multipiler)
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
        self.plot_train_loss(batch, batch_idx, seed=self.seed) # DEFAULT USE SEED 42
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
