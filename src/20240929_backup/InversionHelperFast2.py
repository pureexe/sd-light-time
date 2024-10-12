import torch
import numpy as np 
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from torch.amp import autocast, GradScaler


try:
    import bitsandbytes as bnb
    USE_BITSANDBYTES = True
except:
    USE_BITSANDBYTES = False

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

def get_latent_from_image(vae, image, generator=None):
    """_summary_

    Args:
        vae (_type_): VAE Autoencoder class
        image (_type_): image in format [-1,1]

    Returns:
        _type_: _description_
    """
  
    latents =  vae.encode(image.to(vae.dtype)).latent_dist.sample(generator=generator)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(vae.dtype)
    return latents

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

def controlnet_step(pipe, hidden_states, latents, timestep, callback_kwargs, is_apply_cfg=False):
    guess_mode = callback_kwargs['guess_mode']
    if guess_mode:
        control_model_input = latents
        controlnet_prompt_embeds = hidden_states.chunk(2)[1]
        condition_image = callback_kwargs['image'].chunk(2)[1]
    else:
        if is_apply_cfg:
            control_model_input = torch.cat([latents] * 2)
            condition_image = callback_kwargs['image']
        else:
            control_model_input = latents       
            condition_image = callback_kwargs['image'].chunk(2)[1]  
        controlnet_prompt_embeds = hidden_states


    control_model_input = pipe.scheduler.scale_model_input(control_model_input, timestep)
    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
        control_model_input,
        timestep,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=condition_image,
        conditioning_scale=callback_kwargs['cond_scale'],
        guess_mode=guess_mode,
        return_dict=False,
    )
    if guess_mode:
        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    return down_block_res_samples, mid_block_res_sample



def denoise_step(pipe, hidden_states, latents, timestep, callback_kwargs, noise_pred_text=None, use_guidance=True):

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if noise_pred_text is None and use_guidance  else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

        # support for controlnet
        if 'image' in callback_kwargs:
            down_block_res_samples, mid_block_res_sample = controlnet_step(pipe, hidden_states, latents, timestep, callback_kwargs, is_apply_cfg=noise_pred_text is  None and use_guidance)
            callback_kwargs['down_block_res_samples'] = down_block_res_samples
            callback_kwargs['mid_block_res_sample'] = mid_block_res_sample

        # feed  prompt to unet
        noise_pred = compute_noise(pipe, hidden_states, latent_model_input, timestep, callback_kwargs)

        if use_guidance:
            # classifier free guidance
            if noise_pred_text is None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            else:
                noise_pred_uncond = noise_pred

            # classifier free guidance
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if hasattr(pipe,'guidance_rescale') and pipe.guidance_rescale > 0.0:
                # Rescale noise cfg, Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_scale)

        else:
            noise_pred_uncond = None
            noise_pred_text = noise_pred     
                
        # predict next latents
        predict_latents = pipe.scheduler.step(noise_pred, timestep, latents,  **callback_kwargs['extra_step_kwargs'], return_dict=False)[0]
 
        return predict_latents, noise_pred_text, noise_pred_uncond


def get_ddim_latents(pipe, image, text_embbeding, num_inference_steps, generator = None, controlnet_image=None, guidance_scale=1.0):
    nomral_scheduler = DDIMScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')

    pipe.scheduler = inverse_scheduler

    z0_noise = get_latent_from_image(pipe.vae, image)


    # do ddim inverse to noise 
    ddim_latents = []
    def callback_ddim(pipe, step_index, timestep, callback_kwargs):
        ddim_latents.append(callback_kwargs['latents'])
        return callback_kwargs
        
    ddim_args = {
        "prompt_embeds": text_embbeding,
        "guidance_scale": guidance_scale,
        "latents": z0_noise,
        "output_type": 'latent',
        "return_dict": False,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "callback_on_step_end": callback_ddim
    }
    if controlnet_image is not None:
        ddim_args['image'] = controlnet_image

    zt_noise, _ = pipe(**ddim_args)
    pipe.scheduler = nomral_scheduler
    return ddim_latents

def get_null_embeddings_BAK(pipe, ddim_latents, text_embbeding, negative_embedding, guidance_scale,  num_inference_steps, before_positive_pass_callback=None, before_negative_pass_callback=None,  before_final_denoise_callback=None, before_final_denoise_callback2=None, after_final_denoise_callback=None, controlnet_image=None, num_null_optimization_steps=10, generator=None):
    """
    compute null text embeding for each denoisng step

    Args:
        pipe (_type_): SD Pipeline
        ddim_latents (_type_): VAE latents from ddim. order by low_timestep (less noise) to high timestep (high noise) which is the output of ddim. shape [num_inference_steps, 4, 64, 64]
        text_embbeding (_type_): text embeddings. shape [77, 768]
        negative_embedding (_type_): negative text embeddings. shape [77, 768]
        guidance_scale (_type_): guidance scale
        num_inference_steps (_type_): number of DDIM inversion / null text inversion step_
        controlnet_image (_type_, optional): controlnet image. Defaults to None.
        num_null_optimization_steps (int, optional): number of null text optimization (should same as ddim). Defaults to 10.
        generator (_type_, optional): pytorch random genenrator. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # flip ddim latents
    ddim_latents = ddim_latents[::-1]

    null_embeddings = []
    null_latents = []

    def callback_optimize_nulltext(pipe, step_index, timestep, callback_kwargs, loss_multiplier=100, use_amp=False):

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = negative_prompt_embeds.clone().detach()
        before_text_embbeding = text_embbeding.clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == num_inference_steps:
            callback_kwargs['latents'] = latents
            return callback_kwargs

        if before_positive_pass_callback is not None:
            before_positive_pass_callback()

        _, noise_pred_text, _ = denoise_step(pipe, text_embbeding, latents, timestep, callback_kwargs, noise_pred_text=None, use_guidance=False) #USE_CFG=FALSE

        if before_negative_pass_callback is not None:
            before_negative_pass_callback()

        # backprop to get negative_prompt_embeds
        with torch.enable_grad():                
            negative_prompt_embeds.requires_grad = True
            if USE_BITSANDBYTES:
                optimizer_class = bnb.optim.Adam8bit
            else:
                optimizer_class = torch.optim.Adam

            optimizer = optimizer_class([negative_prompt_embeds], lr=1e-2)

            if use_amp:
                scaler = GradScaler()
    
            for _ in range(num_null_optimization_steps):
                with autocast(device_type='cuda', enabled=use_amp):
                    optimizer.zero_grad()
                    predict_latents, _, _ = denoise_step(
                        pipe=pipe, 
                        hidden_states=negative_prompt_embeds, #[1,77,768]
                        #hidden_states=torch.cat([negative_prompt_embeds, text_embbeding], dim=0),
                        latents=latents,
                        timestep=timestep,
                        callback_kwargs=callback_kwargs,
                        noise_pred_text=noise_pred_text,
                        use_guidance=True
                    ) #USE_CFG=FALSE, NOGATE

                    # calculate loss with next latents
                    loss = torch.nn.functional.mse_loss(predict_latents * loss_multiplier, ddim_latents[step_index+1] * loss_multiplier)
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    if loss < 1e-5 * np.abs(loss_multiplier) : #early stopping mention in the paper
                        break
        if before_final_denoise_callback is not None:
            before_final_denoise_callback()

        # text embeding stay the same.
        # compute noise uncond for final time after all updateded
        predict_latents, noiseText1, noiseUncond1 = denoise_step(
            pipe=pipe, 
            hidden_states=negative_prompt_embeds,
            #hidden_states=torch.cat([negative_prompt_embeds, text_embbeding], dim=0),
            latents=latents,
            noise_pred_text=noise_pred_text, # we reuse the positive noise 
            callback_kwargs=callback_kwargs,
            timestep=timestep,
            use_guidance=True
        ) #USE_CFG=FALSE, NOGATE

        if before_final_denoise_callback2 is not None:
            before_final_denoise_callback2()

        predict_latents2, noiseText2, noiseUncond2 = denoise_step(
            pipe=pipe, 
            #hidden_states=negative_prompt_embeds,
            hidden_states=torch.cat([negative_prompt_embeds, text_embbeding], dim=0),
            latents=latents,
            #noise_pred_text=noise_pred_text, # we reuse the positive noise 
            callback_kwargs=callback_kwargs,
            timestep=timestep,
            use_guidance=True
        ) #USE_CFG=Tue, True
        # check if this close 
        loss = torch.nn.functional.mse_loss(predict_latents, predict_latents2)
        predict_latents2 = predict_latents
        print(f"latents check: {loss.item():.8f}")
        print(f"noiseText close check: {torch.nn.functional.mse_loss(noiseText1, noiseText2).item():.8f}")
        print(f"noiseUncond close check: {torch.nn.functional.mse_loss(noiseUncond1, noiseUncond2).item():.8f}")
        print("=================")
        if after_final_denoise_callback is not None:
            after_final_denoise_callback()  # callback to re-enable light condition

        negative_prompt_embeds = negative_prompt_embeds.detach()
        # print negative_prompt_embeds meean
        callback_kwargs['prompt_embeds'] = torch.cat([negative_prompt_embeds, text_embbeding])
        null_embeddings.append(negative_prompt_embeds)
        callback_kwargs['latents'] = predict_latents.detach()
        null_latents.append(predict_latents)
        return callback_kwargs
    
    # Stable diffusion parameters

    sd_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding, 
        "guidance_scale": guidance_scale,
        "latents": ddim_latents[0],
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": ["latent_model_input", "prompt_embeds", "timestep_cond", "added_cond_kwargs","extra_step_kwargs"],
        "callback_on_step_end": callback_optimize_nulltext
    }
    if controlnet_image is not None:
        sd_args['image'] = controlnet_image
        sd_args['callback_on_step_end_tensor_inputs'] += ['cond_scale', 'guess_mode', 'image']

    # run stable diffusion
    _ = pipe(**sd_args)
    return null_embeddings, null_latents

    
def apply_null_embedding(pipe, latents, null_embeddings, text_embbeding, guidance_scale, num_inference_steps, controlnet_image=None, generator=None, null_latents=None):
    """
    Re-generated image with null embeddings

    Args:
        pipe (_type_): SD pipeline
        latents (_type_): VAE latents from ddim. shape [1, 4, 64, 64]
        null_embeddings (_type_): embeddings from get_null_embedding. shape [num_inference_steps, 77, 768]
        text_embbeding (_type_): text embeddings. shape [77, 768]
        guidance_scale (_type_): guidance scale
        num_inference_steps (_type_): number of inference steps
        controlnet_image (_type_, optional): ControlNet condition image. Defaults to None.
        generator (_type_, optional): random generator for pytorch. Defaults to None.
    """

    def callback_apply_nulltext(pipe, step_index, timestep, callback_kwargs):

        # check if we still can apply null embedding in the next step
        try:
            negative_embedding = null_embeddings[step_index+1]
        except IndexError:
            # we can't predict next latents for the last step, So we interrupt the diffusion process.
            pipe._interrupt = True
            return callback_kwargs
        
        if null_latents is not None:
            loss = torch.nn.functional.mse_loss(callback_kwargs['latents'], null_latents[step_index])
            print(f"Loss: {loss.item():.8f}")

        _, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        callback_kwargs['prompt_embeds'] = torch.cat([negative_embedding, text_embbeding], dim=0)
        return callback_kwargs
    
    pipe_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": null_embeddings[0],
        "latents": latents,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "callback_on_step_end_tensor_inputs": ["latents", "latent_model_input", "prompt_embeds"],
        "callback_on_step_end": callback_apply_nulltext,
        "output_type": "pt",
    }

    if controlnet_image is not None:
        pipe_args['image'] = controlnet_image

    pt_image = pipe(**pipe_args)['images']
    return pt_image
    

 

def null_text_denoising():
    pass