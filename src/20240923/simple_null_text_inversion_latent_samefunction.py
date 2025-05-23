import numpy as np 
import torchvision
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
import torch 
from tqdm.auto import tqdm
import bitsandbytes as bnb


MASTER_TYPE = torch.float32
SEED = 42
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"
#PROMPT = "A dog in the park"
PROMPT = "several pots of plants sit on a counter top"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.0
NULL_STEP = 10
EXP_NAME = f"latentinput_samefunction_{NUM_INFERENCE_STEPS}_3"
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

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

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
    null_timesteps = []
    null_latents = []
    null_predicts = []

    def callback_nulltext(pipe, step_index, timestep, callback_kwargs):

        null_timesteps.append(timestep)

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = negative_prompt_embeds.clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            print(f"Skip: {null_timesteps[step_index-1]} -> {null_timesteps[step_index]}")
            callback_kwargs['latents'] = latents
            return callback_kwargs
        
        # compute noise_pred_text

        noise_pred_text = pipe.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=text_embbeding,
            return_dict=False,
            timestep_cond=callback_kwargs['timestep_cond'],
            cross_attention_kwargs=pipe.cross_attention_kwargs,
            added_cond_kwargs=callback_kwargs['added_cond_kwargs']
        )[0]

        # backpropt to get negative_prompt_embeds
        with torch.enable_grad():                
            negative_prompt_embeds.requires_grad = True
            optimizer = torch.optim.Adam([negative_prompt_embeds], lr=1e-2)
            for _ in range(NULL_STEP):
                optimizer.zero_grad()
                # prepare for input
                unet_kwargs = {
                    "sample": latents,
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "return_dict": False,
                    "timestep_cond": callback_kwargs['timestep_cond'],
                    "cross_attention_kwargs": pipe.cross_attention_kwargs,
                    "added_cond_kwargs": callback_kwargs['added_cond_kwargs']
                }
                # support for controlnet
                if 'down_block_res_samples' in callback_kwargs:
                    unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
                    unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']

                noise_pred_uncond = pipe.unet(**unet_kwargs)[0]
                
                # classifier free guidance
                #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)
                
                # predict next latents
                callback_kwargs['extra_step_kwargs']['generator'] = torch.Generator().manual_seed(SEED)
                predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
                
                # calculate loss with next latents
                loss = torch.nn.functional.mse_loss(predict_latents, ddim_latents[step_index+1])
                loss.backward()
                optimizer.step()
    
                print(f"{loss.item():.6f}")
                if loss < 1e-5: #early stopping mention in the paper
                    break

        # compute noise uncond for final time after all updateded
        noise_pred_uncond = pipe.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=negative_prompt_embeds,
            return_dict=False,
            timestep_cond=callback_kwargs['timestep_cond'],
            cross_attention_kwargs=pipe.cross_attention_kwargs,
            added_cond_kwargs=callback_kwargs['added_cond_kwargs']
        )[0]

        noise_out = torch.cat([noise_pred_uncond, noise_pred_text], dim=0) # this is for later check

        # classifier free guidance
        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)
                
        # predict next latents
        callback_kwargs['extra_step_kwargs']['generator'] = torch.Generator().manual_seed(SEED)
        predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
        

        negative_prompt_embeds = negative_prompt_embeds.detach()
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, text_embbeding])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        null_latents.append(predict_latents)
        null_predicts.append(noise_out)
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
        "callback_on_step_end": callback_nulltext
    }
    output_image = pipe(**sd_args)['images'][0]
    output_image.save(f"output_copyroom10_{EXP_NAME}.jpg")

    def callback_apply_nulltext(pipe, step_index, timestep, callback_kwargs):
                
        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM
        latent_model_input = callback_kwargs['latent_model_input'].clone().detach()

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = null_embeddings[step_index].clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            callback_kwargs['latents'] = latents
            return callback_kwargs
        

        with torch.no_grad():                
            if True:
                # prepare for input
                #embedd = torch.cat([negative_prompt_embeds, text_embbeding], dim=0)
                embedd = text_embbeding
                unet_kwargs = {
                    "sample": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": torch.cat([null_embeddings[step_index], text_embbeding], dim=0),
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
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if torch.allclose(noise_pred_uncond, null_predicts[step_index].chunk(2)[0], atol=1e-5):
                    print("noise_pred_uncond: Match")
                else:
                    print("noise_pred_uncond: Not Match")

                # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)
                
                
                # predict next latents
                callback_kwargs['extra_step_kwargs']['generator'] = torch.Generator().manual_seed(SEED)
                predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
                

        # compare different from null_latents
        loss = torch.nn.functional.mse_loss(predict_latents, null_latents[step_index])
        print(f"Loss: {loss.item():.6f}")

        negative_prompt_embeds = negative_prompt_embeds.detach()
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, text_embbeding])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        null_latents.append(predict_latents)
        return callback_kwargs

    def callback_apply_nulltext_shortver(pipe, step_index, timestep, callback_kwargs):
        # skip if out of index 
        try:
            print(f"Step: {timestep}, Index: {null_timesteps[step_index]}")
            negative_text_embedding = null_embeddings[step_index+1]
        except IndexError:
            print(f"Skip: {null_timesteps[step_index-1]} -> {null_timesteps[step_index]}")
            latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM
            callback_kwargs['latents'] = latents
            return callback_kwargs
        
        callback_kwargs['extra_step_kwargs']['generator'] = torch.Generator().manual_seed(SEED)

        # compute distance between latents and null_latents
        _, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        # apply null text
        callback_kwargs['prompt_embeds'] = torch.cat([negative_text_embedding, text_embbeding])
        return callback_kwargs

    def callback_apply_nulltext_WORK(pipe, step_index, timestep, callback_kwargs):
                
        latent_model_input = callback_kwargs['latent_model_input'].clone().detach()

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        #negative_prompt_embeds = negative_prompt_embeds.clone().detach()
        negative_prompt_embeds = null_embeddings[step_index].clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == NUM_INFERENCE_STEPS:
            callback_kwargs['latents'] = latents
            return callback_kwargs
        
        noise_pred_text = pipe.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=text_embbeding,
            return_dict=False,
            timestep_cond=callback_kwargs['timestep_cond'],
            cross_attention_kwargs=pipe.cross_attention_kwargs,
            added_cond_kwargs=callback_kwargs['added_cond_kwargs']
        )[0]

        # check if noise_pred_text match
        if torch.allclose(noise_pred_text, null_predicts[step_index].chunk(2)[1], atol=1e-5):
            print("noise_pred_text: Match")
        else:
            print("noise_pred_text: Not Match")
        

        with torch.no_grad():                
            if True:
                # prepare for input
                #embedd = torch.cat([negative_prompt_embeds, text_embbeding], dim=0)
                embedd = text_embbeding
                unet_kwargs = {
                    "sample": latents, #latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": null_embeddings[step_index],
                    "return_dict": False,
                    "timestep_cond": callback_kwargs['timestep_cond'],
                    "cross_attention_kwargs": pipe.cross_attention_kwargs,
                    "added_cond_kwargs": callback_kwargs['added_cond_kwargs']
                }
                # support for controlnet
                if 'down_block_res_samples' in callback_kwargs:
                    unet_kwargs['down_block_additional_residuals'] = callback_kwargs['down_block_res_samples']
                    unet_kwargs['mid_block_additional_residual'] = callback_kwargs['mid_block_res_sample']
                
                noise_pred_uncond = pipe.unet(**unet_kwargs)[0]
                # check if later pass match 
                #if torch.allclose(noise_pred.chunk(2)[1], null_predicts[step_index].chunk(2)[1], atol=1e-5):
                if torch.allclose(noise_pred_uncond, null_predicts[step_index].chunk(2)[0], atol=1e-5):
                    print("noise_pred_uncond: Match")
                else:
                    print("noise_pred_uncond: Not Match")

                #noise_pred = null_predicts[step_index]

                # classifier free guidance
                #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)
                
                
                # predict next latents
                callback_kwargs['extra_step_kwargs']['generator'] = torch.Generator().manual_seed(SEED)
                predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
                

        # compare different from null_latents
        loss = torch.nn.functional.mse_loss(predict_latents, null_latents[step_index])
        print(f"Loss: {loss.item():.6f}")

        negative_prompt_embeds = negative_prompt_embeds.detach()
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, text_embbeding])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        null_latents.append(predict_latents)
        return callback_kwargs
    
    pipe_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": null_embeddings[0],
        "latents": copy_zt_noise.clone(),
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latent_model_input", "prompt_embeds", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "noise_pred"],
        "callback_on_step_end": callback_apply_nulltext,
    }
    output_image = pipe(**pipe_args)['images'][0]
    output_image.save(f"output_copyroom10_reinference_{EXP_NAME}.jpg")






if __name__ == "__main__":
    main()