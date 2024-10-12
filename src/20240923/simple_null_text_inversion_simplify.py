import numpy as np 
import torchvision
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
import torch 
from tqdm.auto import tqdm


try:
    import bitsandbytes as bnb
    USE_BITSANDBYTES = True
except:
    USE_BITSANDBYTES = False



MASTER_TYPE = torch.float32

SEED = 42
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"
#PROMPT = "A dog in the park"
PROMPT = "several pots of plants sit on a counter top"
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.0
NULL_STEP = 10
EXP_NAME = f"latentinput_simplify_{NUM_INFERENCE_STEPS}_3"
#INPUT_IMAGE = "src/20240923/copyroom10.png"
INPUT_IMAGE = "copyroom10.png"

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

    # add 
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "latent_model_input", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "noise_pred", "timesteps"]

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
    copy_zt_noise = zt_noise.clone()

    pipe.scheduler = nomral_scheduler

    
    
    # flip list of latents and timesteps
    ddim_latents = ddim_latents[::-1]
    ddim_timesteps = ddim_timesteps[::-1]

    null_embeddings = []
    null_latents = []
    null_predicts = []

    def compute_noise(pipe, hidden_states, latents, timestep, callback_kwargs):
        # unet parameters
        unet_kwargs = {
            "sample": latents,
            "timestep": timestep,
            "encoder_hidden_states": hidden_states,
            "return_dict": False,
            "timestep_cond": callback_kwargs['timestep_cond'],
            "cross_attention_kwargs": pipe.cross_attention_kwargs,
            "added_cond_kwargs": callback_kwargs['added_cond_kwargs']
        }
        # support for controlnet
        if 'down_block_res_samples' in callback_kwargs:
            unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
            unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']
        
        noise_pred = pipe.unet(**unet_kwargs)[0]
        return noise_pred


    def denoise_step(pipe, hidden_states, latents, timestep, callback_kwargs, noise_pred_text=None, return_noise=False):

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if noise_pred_text is None else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        # feed to unet
        noise_pred = compute_noise(pipe, hidden_states, latent_model_input, timestep, callback_kwargs)

        # classifier free guidance
        if noise_pred_text is None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        else:
            noise_pred_uncond = noise_pred

        # classifier free guidance
        noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if pipe.guidance_rescale > 0.0:
            # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)     
                
        # predict next latents
        predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
 
        if return_noise:
            return predict_latents, torch.cat([noise_pred_uncond, noise_pred_text, noise_pred])
        else:
            return predict_latents


    def callback_optimize_nulltext(pipe, step_index, timestep, callback_kwargs):

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = negative_prompt_embeds.clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            callback_kwargs['latents'] = latents
            return callback_kwargs
        
        # compute noise_pred_text for onetime instead of repeatly predict every optimization step
        noise_pred_text = compute_noise(pipe, text_embbeding, latents, timestep, callback_kwargs)

        # backprop to get negative_prompt_embeds
        with torch.enable_grad():                
            negative_prompt_embeds.requires_grad = True
            if USE_BITSANDBYTES:
                optimizer_class = bnb.optim.Adam8bit
            else:
                optimizer_class = torch.optim.Adam

            optimizer = optimizer_class([negative_prompt_embeds], lr=1e-2)
            for _ in range(NULL_STEP):
                optimizer.zero_grad()
                predict_latents = denoise_step(
                    pipe=pipe, 
                    hidden_states=negative_prompt_embeds,
                    latents=latents,
                    noise_pred_text=noise_pred_text,
                    timestep=timestep,
                    callback_kwargs=callback_kwargs,
                )
                # calculate loss with next latents
                loss = torch.nn.functional.mse_loss(predict_latents.float(), ddim_latents[step_index+1])
                loss.backward()
                optimizer.step()
    
                if loss < 1e-5: #early stopping mention in the paper
                    break
        
        # compute noise uncond for final time after all updateded
        predict_latents, noise_out = denoise_step(
            pipe=pipe, 
            hidden_states=torch.cat([negative_prompt_embeds, text_embbeding], dim=0),
            latents=latents,
            timestep=timestep,
            callback_kwargs=callback_kwargs,
            return_noise=True
        )

        null_predicts.append(noise_out)
        negative_prompt_embeds = negative_prompt_embeds.detach()
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, text_embbeding])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        null_latents.append(predict_latents)
        return callback_kwargs
    
    
    # do ddim forward to image
    sd_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding, 
        "guidance_scale": GUIDANCE_SCALE,
        "latents": zt_noise,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latent_model_input", "prompt_embeds", "timestep_cond", "added_cond_kwargs","extra_step_kwargs", "timesteps"],
        "callback_on_step_end": callback_optimize_nulltext
    }
    output_image = pipe(**sd_args)['images'][0]
    output_image.save(f"output_copyroom10_{EXP_NAME}.jpg")

    def callback_apply_nulltext_longver(pipe, step_index, timestep, callback_kwargs):

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            callback_kwargs['latents'] = latents
            return callback_kwargs
                
        _, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        prompt_embeds =  torch.cat([null_embeddings[step_index], text_embbeding], dim=0)
       
        with torch.no_grad():
            predict_latents = denoise_step(
                pipe=pipe, 
                hidden_states=prompt_embeds,
                latents=latents,
                timestep=timestep,
                callback_kwargs=callback_kwargs,
            )

        # compare different from null_latents
        loss = torch.nn.functional.mse_loss(predict_latents, null_latents[step_index])
        print(f"Loss: {loss.item():.6f}")

        callback_kwargs['prompt_embeds'] = prompt_embeds
        callback_kwargs['latents'] = predict_latents.detach()
        return callback_kwargs

    def callback_apply_nulltext(pipe, step_index, timestep, callback_kwargs):

        # we can't predict next latents for the last step
        try:
            negative_embedding = null_embeddings[step_index+1]
        except IndexError:
            latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach()
            if step_index + 1 > len(null_embeddings):
                callback_kwargs['latents'] = latents
            return callback_kwargs

        loss = torch.nn.functional.mse_loss(callback_kwargs['latents'], null_latents[step_index])
        print(f"Loss: {loss.item():.8f}")

        _, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        callback_kwargs['prompt_embeds'] = torch.cat([negative_embedding, text_embbeding], dim=0)
        return callback_kwargs

    
    
    pipe_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": null_embeddings[0],
        "latents": copy_zt_noise.clone(),
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latents", "latent_model_input", "prompt_embeds", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "noise_pred"],
        "callback_on_step_end": callback_apply_nulltext,
        "null_predicts": null_predicts,
        "null_latents": null_latents
    }
    output_image = pipe(**pipe_args)['images'][0]
    output_image.save(f"output_copyroom10_reinference_{EXP_NAME}.jpg")






if __name__ == "__main__":
    main()