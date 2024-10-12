import torch
from AffineConsistancy import AffineConsistancy

class AffineConsistancyVaeCompatible(AffineConsistancy):
    def get_vae_features(self, images, generator=None):
        assert images.shape[1] == 3, "Only support RGB image"
        assert images.shape[2] == 256 and images.shape[3] == 256, "Only support 256x256 image"
        #assert images.min() >= 0.0 and images.max() <= 1.0, "Only support [0, 1] range image"
        with torch.inference_mode():
            # VAE need input in range of [-1,1]
            images = images * 2.0 - 1.0
            emb = self.pipe.vae.encode(images).latent_dist.sample()
            flattened_emb = emb.view(emb.size(0), -1)
            return flattened_emb