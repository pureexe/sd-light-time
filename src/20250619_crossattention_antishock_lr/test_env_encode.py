
import torch
from einops import rearrange
from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset
from constants import DATASET_ROOT_DIR

from env_encoder import EnvEncoder
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder

def _encode_env(env_encoder, env_input, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(env_encoder.parameters()).dtype
        # env_input = {k: v.to(device=device, dtype=dtype) for k, v in env_input.items()}
        multi_frame = (env_input[0].dim() != 4)
        bsz = env_input[0].size(0)

        _env_input = []
        for v in env_input:
            if multi_frame:
                v = rearrange(v, "b f c h w -> (b f) c h w")

            if env_encoder.config.latent_encoder:
                v = self.vae.encode(v).latent_dist.mode() * self.vae.config.scaling_factor

            _env_input.append(v)

        _env_input = torch.cat(_env_input, dim=1) # [BF, C, H, W]
        _env_embeddings = env_encoder(_env_input)

        env_embeddings = []
        for v in _env_embeddings:
            v = v.flatten(2).transpose(1, 2) # [B, N, C]
            if multi_frame:
                v = rearrange(v, "(b f) n c -> b f n c", b=bsz)
                v = v.repeat(num_videos_per_prompt, 1, 1, 1)
            else:
                v = v.repeat(num_videos_per_prompt, 1, 1)

            env_embeddings.append(v)

        if do_classifier_free_guidance:
            env_embeddings = [
                torch.cat([torch.zeros_like(v), v]) for v in env_embeddings
            ]

        return env_embeddings

env_encoder = EnvEncoder.from_pretrained("/pure/t1/project/diffusion-renderer/checkpoints/diffusion_renderer-forward-svd", subfolder="env_encoder") # , torch_dtype=torch.float16
svd_vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid", subfolder="vae") # use SVD vae for Envmap encoder # , varient='fp16', torch_dtype=torch.float16
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae") # use SVD vae for Envmap encoder # , varient='fp16', torch_dtype=torch.float16

train_dataset = DiffusionRendererEnvmapDataset(
    root_dir=DATASET_ROOT_DIR,
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
for batch in train_dataloader:
    features = []
    for k in ['light_ldr', 'light_log_hdr', 'light_dir']:
        print(f"Processing {k}...")
        print(batch[k].shape)
        feature = svd_vae.encode(batch[k]).latent_dist.mode() * svd_vae.config.scaling_factor
        feature_sd = vae.encode(batch[k]).latent_dist.mode() * vae.config.scaling_factor
        print("TOTAL DIFFERNETIAL:", torch.sum((feature - feature_sd)**2))
        features.append(feature) # [B, C, H, W]

    # concatenate features along channel dimension
    env_input = torch.cat(features, dim=1)  # [B, C*3, H, W]
    
    env_features = env_encoder(env_input)

    print("Encoded Environment Features Shape:", len(env_features))
    # show each layer shape
    for i, v in enumerate(env_features):
        print(f"Layer {i} shape: {v.shape}")

    """
    Encoded Environment Features Shape: 4
    Layer 0 shape: torch.Size([1, 320, 64, 64])
    Layer 1 shape: torch.Size([1, 640, 32, 32])
    Layer 2 shape: torch.Size([1, 1280, 16, 16])
    Layer 3 shape: torch.Size([1, 1280, 8, 8])
    """
    
    print("Encoded Environment Embeddings Shape:", [v.shape for v in env_embeddings])
    break  # Just to test the first batch
