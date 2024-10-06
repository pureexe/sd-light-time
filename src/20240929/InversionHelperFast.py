import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from torch.amp import autocast, GradScaler


try:
    import bitsandbytes as bnb
    USE_BITSANDBYTES = True
except:
    USE_BITSANDBYTES = False


def get_null_embeddings(pipe, ddim_latents, text_embbeding, negative_embedding, guidance_scale,  num_inference_steps, controlnet_image=None, num_null_optimization_steps=10, generator=None):
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
    raise NotImplementedError("Need to make proper support method that we condition stabke duffyusin")
    
    # flip ddim latents
    ddim_latents = ddim_latents[::-1]

    null_embeddings = []
    null_latents = []

    def callback_optimize_nulltext(pipe, step_index, timestep, callback_kwargs):

        latents = callback_kwargs['latent_model_input'].chunk(2)[0].clone().detach() # this trick only work for scheduler that don't scale input such as DDIM

        negative_prompt_embeds, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        negative_prompt_embeds = negative_prompt_embeds.clone().detach()

        # we can't predict next latents for the last step
        if step_index+1 == num_inference_steps:
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
            scaler = GradScaler()

            
            for _ in range(num_null_optimization_steps):
                with autocast(device_type='cuda'):
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
                    loss = torch.nn.functional.mse_loss(predict_latents, ddim_latents[step_index+1])
                    #loss.backward()            
                    #optimizer.step()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if loss < 1e-5: #early stopping mention in the paper
                        break
        
        # compute noise uncond for final time after all updateded
        predict_latents = denoise_step(
            pipe=pipe, 
            hidden_states=torch.cat([negative_prompt_embeds, text_embbeding], dim=0),
            latents=latents,
            timestep=timestep,
            callback_kwargs=callback_kwargs,
        )

        negative_prompt_embeds = negative_prompt_embeds.detach()
        # print negative_prompt_embeds meean
        print(f"Mean: {negative_prompt_embeds.mean().item():.8f}")
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

    # run stable diffusion
    _ = pipe(**sd_args)
    return null_embeddings, null_latents


########### VERSION 2 

def get_null_embeddings(pipe, ddim_latents, text_embbeding, negative_embedding, guidance_scale,  num_inference_steps, before_positive_pass_callback=None, before_negative_pass_callback=None,  before_final_denoise_callback=None, after_final_denoise_callback=None, controlnet_image=None, num_null_optimization_steps=10, generator=None):
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

        # we can't predict next latents for the last step
        if step_index+1 == num_inference_steps:
            callback_kwargs['latents'] = latents
            return callback_kwargs

        if before_positive_pass_callback is not None:
            before_positive_pass_callback()

        # support for controlnet for positive prompt
        if 'image' in callback_kwargs:
            down_block_res_samples, mid_block_res_sample = controlnet_step(pipe, text_embbeding, latents, timestep, callback_kwargs, is_apply_cfg=False)
            callback_kwargs['down_block_res_samples'] = down_block_res_samples
            callback_kwargs['mid_block_res_sample'] = mid_block_res_sample

        # compute positive prompt for single time
        noise_pred_text = compute_noise(pipe, text_embbeding, latents, timestep, callback_kwargs)

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
                    predict_latents = denoise_step(
                        pipe=pipe, 
                        hidden_states=negative_prompt_embeds,
                        latents=latents,
                        timestep=timestep,
                        callback_kwargs=callback_kwargs,
                        noise_pred_text=noise_pred_text,
                    )
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
        # compute noise uncond for final time after all updateded
        predict_latents = denoise_step(
            pipe=pipe, 
            hidden_states=negative_prompt_embeds,
            latents=latents,
            noise_pred_text=noise_pred_text, # we reuse the positive noise 
            callback_kwargs=callback_kwargs,
            timestep=timestep,
        )
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