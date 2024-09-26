import numpy as np 
import torchvision
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
import torch 
from tqdm.auto import tqdm
import bitsandbytes as bnb
from torch.cuda.amp import autocast, GradScaler


MASTER_TYPE = torch.float16
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#PROMPT = "A dog in the park"
PROMPT = "several pots of plants sit on a counter top"
NUM_INFERENCE_STEPS = 200
GUIDANCE_SCALE = 7.0
NULL_STEP = 10
EXP_NAME = "v3_200"

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
    INPUT_IMAGE = "src/20240923/copyroom10.png"

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

    
    # flip list of latents and timesteps
    ddim_latents = ddim_latents[::-1]
    ddim_timesteps = ddim_timesteps[::-1]

    null_embeddings = []


    def callback_nulltext(pipe, step_index, timestep, callback_kwargs):
                
        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            return callback_kwargs
        
        latents = callback_kwargs['latents'].clone().detach()
        latent_model_input = latents #torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        negative_prompt_embeds, prompt_embeds = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = negative_prompt_embeds.clone().detach()

        # compute positive unet 
        unet_kwargs = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "return_dict": False,
        }
        # support for controlnet
        if 'down_block_res_samples' in callback_kwargs:
            unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
            unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']

        noise_pred_text = pipe.unet(**unet_kwargs)[0]


        with torch.enable_grad():
            if True:
                negative_prompt_embeds.requires_grad = True
                optimizer = bnb.optim.Adam8bit([negative_prompt_embeds], lr=1e-2)
                for _ in range(NULL_STEP):
                    optimizer.zero_grad()
                    # prepare for input
                    unet_kwargs = {
                        "sample": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "return_dict": False,
                    }
                    # support for controlnet
                    if 'down_block_res_samples' in callback_kwargs:
                        unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
                        unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']

                    noise_pred_uncond = pipe.unet(**unet_kwargs)[0]

                    # classifier free guidance
                    #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
                    
                    # predict next latents
                    predict_latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                    
                    # calculate loss with next latents
                    loss = torch.nn.functional.mse_loss(predict_latents, ddim_latents[step_index+1])
                    loss.backward()
                    optimizer.step()
                             
                    print(f"{loss.item():.6f}")
                    if loss < 1e-5: #early stopping mention in the paper
                        break

        negative_prompt_embeds = negative_prompt_embeds.detach()
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, prompt_embeds])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        return callback_kwargs

    # do ddim forward to image
    sd_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding, 
        "guidance_scale": GUIDANCE_SCALE,
        "latents": zt_noise,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
        "callback_on_step_end": callback_nulltext
    }
    output_image = pipe(**sd_args)['images'][0]
    output_image.save(f"output_copyroom10_{EXP_NAME}.jpg")

    def callback_apply_nulltext(pipe, step_index, timestep, callback_kwargs):
        # skip the last step
        if step_index + 1 >= len(null_embeddings):
            return callback_kwargs

        # apply null text
        callback_kwargs['prompt_embeds'][0] = null_embeddings[step_index][0]
        return callback_kwargs
    
    pipe_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding,
        "latents": zt_noise.clone(),
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
        "callback_on_step_end": callback_apply_nulltext,
    }
    output_image = pipe(**pipe_args)['images'][0]
    output_image.save(f"output_copyroom10_reinference_{EXP_NAME}.jpg")






if __name__ == "__main__":
    main()