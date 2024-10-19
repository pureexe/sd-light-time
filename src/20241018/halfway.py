from diffusers import StableDiffusionPipeline
import torch
import torchvision 

SEED = 42 

@torch.inference_mode()
def get_text_embeddings(pipe, text):
    if isinstance(text, str):
            text = [text]

    tokens = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids.to(pipe.text_encoder.device)
    
    return pipe.text_encoder(tokens).last_hidden_state

@torch.inference_mode()
def get_noise_schedule(scheduler, device, dtype):
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    return alphas, sigmas

@torch.inference_mode()
def unet_single_pass(pipe, z_t, timestep, embedding_target,  alpha_t, sigma_t):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        noise_pred = pipe.unet(z_t, timestep, embedding_target).sample
        pred_z0 = (z_t - sigma_t * noise_pred) / alpha_t
    return pred_z0

@torch.inference_mode()
def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt1 = "several pots of plants sit on a counter top"
    prompt2 = "a photorealistic image"
    
    for timestep in range(0, 1000, 100):
        alphas, sigmas = get_noise_schedule(pipe.scheduler, "cuda", torch.float16)
        timesteps = torch.tensor([timestep]).to("cuda").long().expand(1)
        prompt1_emb = get_text_embeddings(pipe, prompt1)
        prompt2_emb = get_text_embeddings(pipe, prompt2)
        
        # read image.jpg
        image = torchvision.io.read_image("src/20241018/image.jpg").to("cuda", dtype=torch.float16).unsqueeze(0)
        # normalize image to -1, 1
        image = image / 255.0
        image = image * 2 - 1

        # pass through pipe.vae 
        latents = pipe.vae.encode(image).latent_dist.sample().detach()
        latents = latents * pipe.vae.config.scaling_factor

        torch.manual_seed(SEED)
        noise = torch.randn_like(latents, memory_format=torch.contiguous_format).to("cuda", dtype=torch.float16)

        z_t = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        pred_p0 = unet_single_pass(pipe, z_t, timesteps, prompt1_emb, alphas[timestep], sigmas[timestep])
        pred_p1 = unet_single_pass(pipe, z_t, timesteps, prompt2_emb, alphas[timestep], sigmas[timestep])
        
        # decode the latent to image
        image_p0 = pipe.vae.decode(pred_p0 / pipe.vae.config.scaling_factor, return_dict=False)[0].float().cpu()
        image_p1 = pipe.vae.decode(pred_p1 / pipe.vae.config.scaling_factor, return_dict=False)[0].float().cpu()
        
        # rescale image to 0, 1
        image_p0 = (image_p0 + 1) / 2
        image_p1 = (image_p1 + 1) / 2

        # save image 
        torchvision.utils.save_image(image_p0, f"src/20241018/output/image_p0_timestep{timestep}.png")
        torchvision.utils.save_image(image_p1, f"src/20241018/output/image_p1_timestep{timestep}.png")


 
if __name__ == "__main__":
    main()