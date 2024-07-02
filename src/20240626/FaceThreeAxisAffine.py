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
from FaceThreeAxisDataset import FaceThreeAxisDataset


import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
args = parser.parse_args()
 
 
class FaceThreeAxisAffine(L.LightningModule):
    def __init__(self, learning_rate=1e-3, face100_every=20, guidance_scale=7.5, *args, **kwargs) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.face100_every = face100_every
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        MASTER_TYPE = torch.float16

        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None, torch_dtype=MASTER_TYPE)

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        
        add_light_block(self.pipe.unet, in_channel=3)
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
        set_light_direction(self.pipe.unet, batch['light'], is_apply_cfg=False) #B,C

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    def generate_video_light(self):
        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        # generate 8 face with 32 frame
        TOTAL_FACE = 24
        PER_FACE = TOTAL_FACE // 3
        for face_id in range(TOTAL_FACE):
            print("GENERATING FACE ID: ", face_id)
            if self.use_set_guidance_scale:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/g{self.guidance_scale:.2f}/{face_id}"
            else:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{face_id}"
            os.makedirs(output_dir, exist_ok=True)
            VID_FRAME = 24
            VID_BATCH = 4
            output_frames = []
            direction_mode = face_id // PER_FACE
            directions = torch.linspace(-1, 1, VID_FRAME)[..., None] #[b,1]

            # create 3 axis control seperately
            directions = directions.repeat(1, 3) #B,3
            new_directions = torch.zeros_like(directions)
            new_directions[:, direction_mode] = directions[:, direction_mode]
            directions = new_directions

            need_cfg = self.guidance_scale > 1
            for vid_batch_id in range(VID_FRAME // VID_BATCH):
                set_light_direction(self.pipe.unet, directions[VID_BATCH*vid_batch_id:VID_BATCH*(vid_batch_id+1)], is_apply_cfg=need_cfg)
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

    def generate_video_light_circle(self):
        prompts = []
        with open(os.path.join(DATASET_ROOT_DIR, "prompts.json")) as f:
            prompts = json.load(f)
        # generate 8 face with 32 frame
        TOTAL_FACE = 24
        PER_FACE = TOTAL_FACE // 3
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
            direction_mode = face_id // PER_FACE
            #directions = torch.linspace(-1, 1, VID_FRAME)[..., None] #[b,1]
            directions = torch.linspace(0, 1, VID_FRAME)[..., None] #[b,1]

            # create 3 axis control seperately
            directions = directions.repeat(1, 3) #B,3
            new_directions = torch.zeros_like(directions)
            #new_directions[:, direction_mode] = directions[:, direction_mode]
            new_directions[:, 0] = torch.cos(directions[:, 0] * 2 * np.pi)
            new_directions[:, 1] = torch.sin(directions[:, 1] * 2 * np.pi)
            print(new_directions)
            directions = new_directions

            need_cfg = self.guidance_scale > 1
            for vid_batch_id in range(VID_FRAME // VID_BATCH):
                set_light_direction(self.pipe.unet, directions[VID_BATCH*vid_batch_id:VID_BATCH*(vid_batch_id+1)], is_apply_cfg=need_cfg)
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
        
    def generate_tensorboard_guidance(self, batch, batch_idx):
        set_light_direction(self.pipe.unet, batch['light'], is_apply_cfg=True)
        for guidance_scale in np.arange(1,11,0.5):
            pt_image, _ = self.pipe(
                batch['text'], 
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=50,
                return_dict = False,
                generator=torch.Generator().manual_seed(42)
            )
            gt_image = (batch["pixel_values"] + 1.0) / 2.0
            images = torch.cat([gt_image, pt_image], dim=0)
            image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
            self.logger.experiment.add_image(f'guidance_scale/{guidance_scale:0.2f}', image, self.global_step)
        
    def test_step(self, batch, batch_idx):
        #self.generate_video_light()
        self.generate_video_light_circle()


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and (self.current_epoch+1) % self.face100_every == 0 and self.current_epoch > 1 :
            self.generate_video_light()

        assert (batch['light'] <= 1.0).all() and (batch['light'] >=-1.0).all()  # currently support only left and right
        
        self.generate_tensorboard(batch, batch_idx)
        return torch.zeros(1, )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale
