import os 
import torch 
import torchvision
import lightning as L
import numpy as np
import torchvision
import json
from tqdm.auto import tqdm

from constants import *
from diffusers import StableDiffusionPipeline
from LightEmbedingBlock import set_light_direction, add_light_block
from FaceSingleAxisDataset import FaceSingleAxisDataset


import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
args = parser.parse_args()
 
 
class FaceSingleAxisAffine(L.LightningModule):
    def __init__(
            self, 
            learning_rate=1e-3, 
            face100_every=20, 
            guidance_scale=7.5, 
            viz_video=True,
            viz_image=False,
            *args, 
            **kwargs
        ) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.face100_every = face100_every
        self.learning_rate = learning_rate
        self.viz_video = viz_video
        self.viz_image = viz_image
        self.save_hyperparameters()
        MASTER_TYPE = torch.float16


        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None, torch_dtype=MASTER_TYPE)
        #self.pipe.safety_checker = None
        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        add_light_block(self.pipe.unet)        
        self.pipe.to('cuda')
        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.unet_trainable = torch.nn.ParameterList(unet_trainable)


    def training_step(self, batch, batch_idx):

        text_inputs = self.pipe.tokenizer(
                batch['text'],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids


        latents = self.pipe.vae.encode(batch['pixel_values']).latent_dist.sample().detach()

        latents = latents * self.pipe.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long().to(latents.device)

        text_input_ids = text_input_ids.to(latents.device)
        encoder_hidden_states = self.pipe.text_encoder(text_input_ids)[0]

        # 
        target = noise 
    
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # set light direction
        assert torch.logical_and(batch['light'] <= 1.0, batch['light'] >= -1.0).all()  
        #assert batch['light'][0] <= 1.0 and batch['light'][0]>=-1.0  # currently support [-1,1]
        #assert len(batch['light']) == 1 #current support only batch size = 1
        #self.pipe.unet.set_direction(batch['light'][0]) 
        set_light_direction(self.pipe.unet, batch['light'], is_apply_cfg=False)

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    def generate_video_light(self):
        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        # generate 8 face with 32 frame
        for face_id in range(8):
            print("GENERATING FACE ID: ", face_id)
            if self.use_set_guidance_scale:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/g{self.guidance_scale:.2f}/{face_id}"
            else:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{face_id}"
            os.makedirs(output_dir, exist_ok=True)
            VID_FRAME = 24
            VID_BATCH = 4
            output_frames = []
            directions = torch.linspace(-1, 1, VID_FRAME)[..., None] #[b,1]
            for vid_batch_id in range(VID_FRAME // VID_BATCH):
                set_light_direction(self.pipe.unet, directions[VID_BATCH*vid_batch_id:VID_BATCH*(vid_batch_id+1)], is_apply_cfg=True)
                image, _ = self.pipe(
                    prompts[f"{face_id:05d}"],
                    num_images_per_prompt=VID_BATCH,
                    output_type="pil",
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=50,
                    return_dict = False,
                    generator=[torch.Generator().manual_seed(42) for _ in range(VID_BATCH)]
                )
                for frame_id in range(VID_BATCH):
                    image[frame_id].save(f"{output_dir}/{(VID_BATCH*vid_batch_id) + frame_id:03d}.png")
                    output_frames.append(torchvision.transforms.functional.pil_to_tensor(image[frame_id])[None,None]) #B,T,C,H,W
            output_frames = torch.cat(output_frames, dim=1)
            self.logger.experiment.add_video(f'face/{face_id:03d}', output_frames, self.global_step, fps=6)
        
    def generate_tensorboard(self, batch, batch_idx):
        set_light_direction(self.pipe.unet, batch['light'], is_apply_cfg=True)
        pt_image, _ = self.pipe(
            batch['text'], 
            output_type="pt",
            guidance_scale=self.guidance_scale,
            num_inference_steps=50,
            return_dict = False,
            generator=torch.Generator().manual_seed(42)
        )
        gt_image = (batch["pixel_values"] + 1.0) / 2.0
        images = torch.cat([gt_image, pt_image], dim=0)
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)

    def generate_image_light(self):
        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        for light_name, light_direction in zip(['left','right'], [-1.0, 1.0]):
            if not self.use_set_guidance_scale:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{light_name}"
            else:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/guidance_{self.guidance_scale}/{light_name}"
            os.makedirs(output_dir, exist_ok=True)
            set_light_direction(self.pipe.unet, torch.tensor([light_direction]), is_apply_cfg=True)
            for seed in range(100):
                image, _ = self.pipe(
                    prompts[f"{seed:05d}"], 
                    output_type="pil",
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=50,
                    return_dict = False,
                    generator=torch.Generator().manual_seed(seed)
                )
                image[0].save(f"{output_dir}/{seed:03d}.png")

    def test_step(self, batch, batch_idx):
        if self.viz_video:
            self.generate_video_light()
        if self.viz_image:
            self.generate_image_light()

    def validation_step(self, batch, batch_idx):
        #self.generate_image_light() #need to disable soon
        if batch_idx == 0 and (self.current_epoch+1) % self.face100_every == 0 and self.current_epoch > 1 :
            if self.viz_video:
                self.generate_video_light()
            if self.viz_image:
                self.generate_image_light()

        assert batch['light'][0] <= 1.0 and batch['light'][0]>=-1.0  # currently support only left and right
        assert len(batch['light']) == 1 #current support only batch size = 1
        
        self.generate_tensorboard(batch, batch_idx)
        return torch.zeros(1, )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale
