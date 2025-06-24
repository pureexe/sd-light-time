
import torch
from einops import rearrange
from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset
from constants import DATASET_ROOT_DIR

from env_encoder import EnvEncoder
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from safetensors.torch import save_file, load_file
import torch
import os 
from tqdm.auto import tqdm

#OUTPUT_PATH = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/train/envmap_feature"
OUTPUT_PATH = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/test/envmap_feature"

@torch.inference_mode()
def main():
    env_encoder = EnvEncoder.from_pretrained("/pure/t1/project/diffusion-renderer/checkpoints/diffusion_renderer-forward-svd", subfolder="env_encoder").to('cuda') # , torch_dtype=torch.float16
    svd_vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid", subfolder="vae").to('cuda') # use SVD vae for Envmap encoder # , varient='fp16', torch_dtype=torch.float16
   
    train_dataset = DiffusionRendererEnvmapDataset(
        #root_dir=DATASET_ROOT_DIR,
        root_dir="/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/test",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    for batch in tqdm(train_dataloader):
        scene, filename = batch['name'][0].split('/')
        output_dir = os.path.join(OUTPUT_PATH, scene)
        output_file = os.path.join(output_dir, f"{filename}.safetensors")
        if os.path.exists(output_file):
            continue
    
        features = []
        for k in ['light_ldr', 'light_log_hdr', 'light_dir']:
            feature = svd_vae.encode(batch[k].to('cuda')).latent_dist.mode() * svd_vae.config.scaling_factor
            features.append(feature) # [B, C, H, W]

        # concatenate features along channel dimension
        env_input = torch.cat(features, dim=1)  # [B, C*3, H, W]
        
        env_features = env_encoder(env_input)

        # show each layer shape
        # for i, v in enumerate(env_features):
        #     print(f"Layer {i} shape: {v.shape}")

        output = {}
        for i, v in enumerate(env_features):
            output[f"layer_{i}"] = v[0].to('cpu')
        # save output to file
        os.makedirs(output_dir, exist_ok=True)        
        save_file(output, output_file)

        """
        Encoded Environment Features Shape: 4
        Layer 0 shape: torch.Size([1, 320, 64, 64])
        Layer 1 shape: torch.Size([1, 640, 32, 32])
        Layer 2 shape: torch.Size([1, 1280, 16, 16])
        Layer 3 shape: torch.Size([1, 1280, 8, 8])
        """

if __name__ == "__main__":
    main()