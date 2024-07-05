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
from diffusers import StableDiffusionPipeline

from LightEmbedingBlock import set_light_direction, add_light_block
from EnvmapAffineDataset import EnvmapAffineDataset, log_map_to_range


import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-em', '--envmap_embedder', type=str, default="dino2")
args = parser.parse_args()
 
 
class EnvmapAffine(L.LightningModule):
    def __init__(self, learning_rate=1e-3, face100_every=20, guidance_scale=7.5, envmap_embedder="dino2", *args, **kwargs) -> None:
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
    
        # load envmap_embedder
        self.envmap_embbeder_name = envmap_embedder
        if envmap_embedder == "dino2":
            mlp_in_channel = 768*2
            from transformers import AutoImageProcessor, AutoModel
            dino_path = "facebook/dinov2-base"
            self.envmap_processor = AutoImageProcessor.from_pretrained(dino_path)
            self.envmap_embbeder = [AutoModel.from_pretrained(dino_path)]
            self.envmap_embbeder[0].requires_grad_(False)
            self.envmap_embbeder[0].to('cuda')
        elif envmap_embedder == "slimnet":
            mlp_in_channel = 1024*2
            self.envmap_embbeder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),  
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1), 
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=2, padding=1),
            )
        elif envmap_embedder == "clip":
            # load clip image encoder 
            mlp_in_channel = 1024*2
            from transformers import AutoProcessor, CLIPVisionModel
            clip_path = "openai/clip-vit-large-patch14"
            self.envmap_embbeder = [CLIPVisionModel.from_pretrained(clip_path, torch_dtype=MASTER_TYPE)]
            self.envmap_processor = AutoProcessor.from_pretrained(clip_path)
            self.envmap_embbeder[0].requires_grad_(False)
            self.envmap_embbeder[0].to('cuda')
        elif envmap_embedder == "vae":
            mlp_in_channel = 32*32*4*2
        else:
            raise NotImplementedError("Not support {envmap_embedder} yet")

        #  add light block to unet, 1024 is the shape of output of both LDR and HDR_Normalized clip combine
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')


        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.unet_trainable = torch.nn.ParameterList(unet_trainable)
    
    def get_clip_features(self, images):
        with torch.inference_mode():
            inputs = self.envmap_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(images.device).to(images.dtype) for k, v in inputs.items()}
            outputs = self.envmap_embbeder[0](**inputs)
            return outputs.pooler_output
    
    def get_slimnet_features(self, images):
        emb =  self.envmap_embbeder(images)
        flattened_emb = emb.view(emb.size(0), -1)
        return flattened_emb

    def get_vae_features(self, images):
        assert images.shape[1] == 3, "Only support RGB image"
        assert images.shape[2] == 256 and images.shape[3] == 256, "Only support 256x256 image"
        assert images.min() >= 0.0 and images.max() <= 1.0, "Only support [0, 1] range image"
        with torch.inference_mode():
            # VAE need input in range of [-1,1]
            images = images * 2.0 - 1.0
            emb = self.pipe.vae.encode(images).latent_dist.sample()
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb

    def get_embedder_features(self, images):
        if self.envmap_embbeder_name in ["clip","dino2"]:
            return self.get_clip_features(images) #CLIP and Dino2 HuggingFace version is compatible
        elif self.envmap_embbeder_name == "slimnet":
            return self.get_slimnet_features(images)
        elif self.envmap_embbeder_name == "vae":
            return self.get_vae_features(images)
        else:
            raise NotImplementedError("Not support {self.envmap_embbeder_name} yet")
        
    def get_light_features(self, ldr_images, normalized_hdr_images):
        ldr_features = self.get_embedder_features(ldr_images)
        hdr_features = self.get_embedder_features(normalized_hdr_images)
        # concat ldr and hdr features
        return torch.cat([ldr_features, hdr_features], dim=-1)

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
        light_features = self.get_light_features(batch['ldr_envmap'],batch['normalized_hdr_envmap'])
        set_light_direction(self.pipe.unet, light_features, is_apply_cfg=False) #B,C

        model_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

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
                    generator=[torch.Generator().manual_seed(42) for _ in range(VID_BATCH)]
                )
                for frame_id in range(VID_BATCH):
                    image[frame_id].save(f"{output_dir}/{(VID_BATCH*vid_batch_id) + frame_id:03d}.png")
                    output_frames.append(torchvision.transforms.functional.pil_to_tensor(image[frame_id])[None,None]) #B,T,C,H,W
            output_frames = torch.cat(output_frames, dim=1)
            self.logger.experiment.add_video(f'face/{face_id:03d}', output_frames, self.global_step, fps=6)
        
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        set_light_direction(self.pipe.unet, self.get_light_features(batch['ldr_envmap'],batch['normalized_hdr_envmap']), is_apply_cfg=True)
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
        if is_save_image:
            os.makedirs(f"{self.logger.log_dir}/rendered_image", exist_ok=True)
            torchvision.utils.save_image(image, f"{self.logger.log_dir}/rendered_image/{batch['name'][0]}_{batch['word_name'][0]}.png")
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
            self.logger.experiment.add_text('envmap_embedder', str(self.envmap_embbeder_name), self.global_step)
        
    def generate_tensorboard_guidance(self, batch, batch_idx):
        raise NotImplementedError("Not implemented yet")
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
        self.generate_tensorboard(batch, batch_idx, is_save_image=True)


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and (self.current_epoch+1) % self.face100_every == 0 and self.current_epoch > 1 :
            self.generate_video_light()
        
        self.generate_tensorboard(batch, batch_idx)
        return torch.zeros(1, )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale
