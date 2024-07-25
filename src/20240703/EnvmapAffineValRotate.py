import torch 
import numpy as np
from envmap import EnvironmentMap, rotation_matrix
from LightEmbedingBlock import set_light_direction, add_light_block
from EnvmapAffine import EnvmapAffine
import torchvision 
import os

def rotate3axis(image, axis_name, rotate_ratio):
    device = image.device
    dtype = image.dtype
    envmap = EnvironmentMap(256, format_='latlong')
    wide = torchvision.transforms.functional.resize(image, (256, 512))
    wide = wide.permute(1, 2, 0).cpu().numpy()
    envmap.data = wide 
    if axis_name == 'x':
        dcm = rotation_matrix(azimuth=rotate_ratio*np.pi*2, elevation=0,roll=0)
    elif axis_name == 'y':
        dcm = rotation_matrix(azimuth=0, elevation=rotate_ratio*np.pi*2,roll=0)
    elif axis_name == 'z':
        dcm = rotation_matrix(azimuth=0, elevation=0,roll=rotate_ratio*np.pi*2)
    envmap.rotate(dcm)
    envmap = envmap.data
    envmap = torch.tensor(envmap).permute(2, 0, 1)
    envmap = torchvision.transforms.functional.resize(envmap, (256, 256)) 
    return envmap.to(device=device, dtype=dtype)


class EnvmapAffineValRotate(EnvmapAffine):


    def generate_rotate3axis(self,batch, batch_idx):
        VID_FRAME = 48
        VID_BATCH = 4
        
        need_cfg = self.guidance_scale > 1
        for axis_name in ['x', 'y', 'z']:
            output_frames = []
            output_ldr_frames = []
            output_dir = f"{self.logger.log_dir}/{batch['name'][0]}_{batch['word_name'][0]}/{axis_name}/"
            output_env_dir = f"{self.logger.log_dir}/{batch['name'][0]}_{batch['word_name'][0]}/{axis_name}_env/"
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_env_dir, exist_ok=True)
            for vid_batch_id in range(VID_FRAME // VID_BATCH):
                # rotate axis i
                rolled_hdr = []
                rolled_ldr = []
                for frame_id in range(VID_BATCH):
                    roll_ratio = (vid_batch_id * VID_BATCH + frame_id) / VID_FRAME
                    ldr = rotate3axis(batch['ldr_envmap'][0], axis_name, roll_ratio)
                    hdr = rotate3axis(batch['normalized_hdr_envmap'][0], axis_name, roll_ratio)
                    output_ldr_frames.append(ldr[None][None])
                    torchvision.utils.save_image(ldr, f"{output_env_dir}/{(VID_BATCH*vid_batch_id) + frame_id:03d}.png")
                    rolled_ldr.append(ldr)
                    rolled_hdr.append(hdr)
                # convert rolled_hdr, rolled_ldr to tensor
                rolled_ldr = torch.stack(rolled_ldr)
                rolled_hdr = torch.stack(rolled_hdr)
                light_features = self.get_light_features(
                    rolled_ldr,
                    rolled_hdr,
                )
                set_light_direction(self.pipe.unet, light_features, is_apply_cfg=need_cfg)
                image, _ = self.pipe(
                    batch['text'][0],
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
            #self.logger.experiment.add_video(f'{batch['name'][0]}_{batch['word_name'][0]}/{axis_name}', output_frames, self.global_step, fps=6)
            self.logger.experiment.add_video(f'{batch["name"][0]}_{batch["word_name"][0]}/{axis_name}', output_frames, self.global_step, fps=6)
            output_ldr_frames = torch.cat(output_ldr_frames, dim=1)
            self.logger.experiment.add_video(f'{batch["name"][0]}_{batch["word_name"][0]}/{axis_name}', output_ldr_frames, self.global_step, fps=6)

    def test_step(self, batch, batch_idx):
        self.generate_rotate3axis(batch, batch_idx)
