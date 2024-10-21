import torch

latents = torch.load('/pure/c2/project/sd-light-time/output/20241020/multi_mlp_fit/lightning_logs/version_0/epoch_0000/ddim_latents/everett_dining1-dir_0_mip2.pt')

for idx in range(len(latents)):
    latent = latents[idx].float().cpu()
    # print latent min max 
    print("latent min: ", latent.min(), "latent max: ", latent.max())