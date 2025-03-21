from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from EnvmapAffineDataset import EnvmapAffineDataset, log_map_to_range
from LightEmbedingBlock import add_light_block, set_light_direction
from transformers import pipeline as transformers_pipeline
import numpy as np
import torch
import torchvision
import ezexr
import lightning as L
import json
from PIL import Image

from constants import *

MASTER_TYPE = torch.float16

class EnvMapControlAffine(L.LightningModule):

    def __init__(self, learning_rate=1e-3, face100_every=20, guidance_scale=7.5, envmap_embedder="vae", *args, **kwargs) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.use_set_guidance_scale = False
        self.face100_every = face100_every
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        

        # load pipeline
        sd_path = "runwayml/stable-diffusion-v1-5"
        controlnet_path = "lllyasviel/sd-controlnet-depth"
        depth_path = "Intel/dpt-hybrid-midas"
        self.depth_estimator = transformers_pipeline("depth-estimation", model=depth_path, device=torch.device("cpu"), dtype=torch.float32)
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=MASTER_TYPE)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path, 
            safety_checker=None, 
            controlnet=controlnet, 
            torch_dtype=MASTER_TYPE
        )

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
    
        # load envmap_embedder
        self.envmap_embbeder_name = envmap_embedder
        if envmap_embedder == "vae":
            mlp_in_channel = 32*32*4*2
        else:
            raise NotImplementedError("Not support {envmap_embedder} yet")

        #  add light block to unet, 1024 is the shape of output of both LDR and HDR_Normalized clip combine
        add_light_block(self.pipe.unet, in_channel=mlp_in_channel)
        self.pipe.to('cuda')

        # filter trainable layer
        unet_trainable = filter(lambda p: p.requires_grad, self.pipe.unet.parameters())
        self.unet_trainable = torch.nn.ParameterList(unet_trainable)

    def get_face_image(self, face_id):
        FACE_PATH = "datasets/face/face2000_single/images/{}/{}.png"
        dir_id = face_id // 1000
        face_path = FACE_PATH.format(f"{dir_id:05d}", f"{face_id:05d}")
        img = Image.open(face_path).convert('RGB').resize((512,512))
        return img

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
            # load face image from id 
            face_image = self.get_face_image(face_id)
            control_image = estimate_scene_depth(face_image, depth_estimator=self.depth_estimator)


            if self.use_set_guidance_scale:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/g{self.guidance_scale:.2f}/{face_id}"
            else:
                output_dir = f"{self.logger.log_dir}/face/step{self.global_step:06d}/{face_id}"
            os.makedirs(output_dir, exist_ok=True)
            control_image.save(f"{output_dir}_control.png")
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
                    generator=[torch.Generator().manual_seed(42) for _ in range(VID_BATCH)],
                    image=control_image
                )
                for frame_id in range(VID_BATCH):
                    image[frame_id].save(f"{output_dir}/{(VID_BATCH*vid_batch_id) + frame_id:03d}.png")
                    output_frames.append(torchvision.transforms.functional.pil_to_tensor(image[frame_id])[None,None]) #B,T,C,H,W
            output_frames = torch.cat(output_frames, dim=1)
            self.logger.experiment.add_video(f'face/{face_id:03d}', output_frames, self.global_step, fps=6)

    def test_step(self, batch, batch_idx):
        self.generate_axis6()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def set_guidance_scale(self, guidance_scale):
        self.use_set_guidance_scale = True
        self.guidance_scale = guidance_scale

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
    
    def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
        set_light_direction(self.pipe.unet, self.get_light_features(batch['ldr_envmap'],batch['normalized_hdr_envmap']), is_apply_cfg=True)
        face_image = (batch["pixel_values"] + 1.0) / 2.0
        face_image = torchvision.transforms.functional.to_pil_image(face_image[0])
        control_image = estimate_scene_depth(face_image, depth_estimator=self.depth_estimator)
        pt_image, _ = self.pipe(
            batch['text'], 
            output_type="pt",
            guidance_scale=self.guidance_scale,
            num_inference_steps=50,
            return_dict = False,
            generator=torch.Generator().manual_seed(42),
            image = control_image
        )
        gt_image = (batch["pixel_values"] + 1.0) / 2.0
        images = torch.cat([gt_image, pt_image], dim=0)
        image = torchvision.utils.make_grid(images, nrow=2, normalize=True)
        self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
        if is_save_image:
            append_path = '' if is_save_image == True else f'{self.current_epoch:06d}' + '/' + is_save_image
            os.makedirs(f"{self.logger.log_dir}/rendered_image/{append_path}", exist_ok=True)
            torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/rendered_image/{append_path}/{batch['name'][0]}_{batch['word_name'][0]}.png")
        if self.global_step == 0 and batch_idx == 0:
            self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
            #self.logger.experiment.add_text('params', str(args), self.global_step)
            self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
            self.logger.experiment.add_text('envmap_embedder', str(self.envmap_embbeder_name), self.global_step)
    
    def generate_axis6(self):
        VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']
        for val_file in VAL_FILES:
            val_dataset = EnvmapAffineDataset(split="0:10", specific_file=""+val_file+".json", dataset_multiplier=10, val_hold=0)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
            for batch_idx, batch in enumerate(val_dataloader):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to('cuda').to(MASTER_TYPE)

                self.generate_tensorboard(batch, batch_idx, is_save_image=val_file)

def estimate_scene_depth(image, depth_estimator):
    depth_map = depth_estimator(images=image)['predicted_depth']
    W, H = image.size
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image