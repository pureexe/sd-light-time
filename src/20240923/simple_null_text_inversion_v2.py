import numpy as np 
import skimage.io
import torchvision
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
import torch 
from tqdm.auto import tqdm
import bitsandbytes as bnb
import skimage

MASTER_TYPE = torch.float32
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "A dog in the park"
NUM_INFERENCE_STEPS = 200
GUIDANCE_SCALE = 7.0
NULL_STEP = 10

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


def main():
    INPUT_IMAGE = "src/20240923/input_dog.jpg"

    # read image usign torchvision
    image = Image.open(INPUT_IMAGE)
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    # resize image to 512x512
    image = torchvision.transforms.Resize((512, 512))(image)
    # rescale image to [-1,1] from [0,1]
    image = image * 2 - 1

    # load pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=MASTER_TYPE,safety_checker=None).to(DEVICE)
    nomral_scheduler = DDIMScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')

    # prepare inputs
    z0_noise = pipe.vae.encode(image.to(DEVICE).to(MASTER_TYPE)).latent_dist.sample(generator=torch.Generator().manual_seed(SEED)) * pipe.vae.config.scaling_factor
    text_embbeding = get_text_embeddings(pipe, PROMPT)
    negative_embedding = get_text_embeddings(pipe, "")

    # do ddim inverse to noise 
    ddim_latents = []
    ddim_timesteps = []
    pipe.scheduler = inverse_scheduler
    def callback_ddim(pipe, step_index, timestep, callback_kwargs):
        ddim_timesteps.append(timestep)
        ddim_latents.append(callback_kwargs['latents'].clone())
        return callback_kwargs

    ddim_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding, 
        "guidance_scale": 1.0,
        "latents": z0_noise,
        "output_type": 'latent',
        "return_dict": False,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end": callback_ddim
    }
    zt_noise, _ = pipe(**ddim_args)

    pipe.scheduler = nomral_scheduler
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)

    
    # flip list of latents and timesteps
    ddim_latents = ddim_latents[::-1]
    ddim_timesteps = ddim_timesteps[::-1]

    null_embeddings = []

    latents = ddim_latents[0].clone()
    negative_prompt_embeds = negative_embedding.clone().detach()
    
    
    with torch.enable_grad():
        negative_prompt_embeds.requires_grad = True
        optimizer = bnb.optim.Adam8bit([negative_prompt_embeds], lr=1e-2)
        for step_index, timestep in enumerate(tqdm(ddim_timesteps)):
            latents = latents.clone().detach()
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

                
            for _ in range(NULL_STEP):
                optimizer.zero_grad()
                # prepare for input
                embedd = torch.cat([negative_prompt_embeds, text_embbeding], dim=0)
                unet_kwargs = {
                    "sample": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": embedd,
                    "return_dict": False,
                }
                # TODO: support for controlnet
                # if 'down_block_res_samples' in callback_kwargs:
                #     unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
                #     unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']

                noise_pred = pipe.unet(**unet_kwargs)[0]

                # classifier free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

                # predict next latents
                predict_latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

                # we can't predict next latents for the last step
                if step_index+1 == NUM_INFERENCE_STEPS:
                    break
                
                print(f"Step: {step_index+1}/{NUM_INFERENCE_STEPS}")
                # calculate loss with next latents
                loss = torch.nn.functional.mse_loss(predict_latents, ddim_latents[step_index+1])
                print(f"{loss.item():0.6f}")
                loss.backward()
                optimizer.step()

                if loss < 1e-5: #early stopping mention in the paper
                    break

            latents = predict_latents

    output_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor)[0]
    output_image = (output_image + 1) / 2
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = skimage.img_as_ubyte(image[0])
    # save image
    skimage.io.imsave("output_dog_null_text_v2.jpg", output_image)
    #output_image.save("output_dog_null_text.jpg")






if __name__ == "__main__":
    main()