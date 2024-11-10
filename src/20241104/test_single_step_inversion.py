import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

import os
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from torchvision import transforms

DTYPE = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEP = 999


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.inference_mode()
def decode_imgs(latents, pipeline, output_type="pt"):
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type=output_type)
    return imgs

@torch.inference_mode()
def encode_imgs(imgs, pipeline, DTYPE):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=DTYPE)
    return latents

def generate_eta_values(
    timesteps, 
    start_step, 
    end_step, 
    eta, 
    eta_trend,
):
    assert start_step < end_step and start_step >= 0 and end_step <= len(timesteps), "Invalid start_step and end_step"
    # timesteps are monotonically decreasing, from 1.0 to 0.0
    eta_values = [0.0] * len(timesteps)
    
    if eta_trend == 'constant':
        for i in range(start_step, end_step):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[start_step] - timesteps[i]) / total_time
    elif eta_trend == 'linear_decrease':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[i] - timesteps[end_step - 1]) / total_time
    else:
        raise NotImplementedError(f"Unsupported eta_trend: {eta_trend}")
    
    return eta_values

@torch.inference_mode()
def interpolated_denoise(
    pipeline, 
    img_latents,
    eta_base,                    # base eta value
    eta_trend,                   # constant, linear_increase, linear_decrease
    start_step,                  # 0-based indexing, closed interval
    end_step,                    # 0-based indexing, open interval
    inversed_latents,            # can be none if not using inversed latents
    use_inversed_latents=True,
    guidance_scale=3.5,
    prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
    num_steps=28,
    seed=42,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None
):
    # using FlowMatchEulerDiscreteScheduler
    pipeline.scheduler.set_timesteps(num_steps, device=pipeline.device)

    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_steps, pipeline.device)
    print("DENOSING PAIR: ", timesteps[-2:])

    # Getting text embedning
    # (
    #     prompt_embeds,
    #     negative_prompt_embeds,
    #     pooled_prompt_embeds,
    #     negative_pooled_prompt_embeds
    # ) = pipeline.encode_prompt(
    #     prompt=prompt, 
    #     prompt_2=prompt,
    #     prompt_3=prompt
    # )

    if use_inversed_latents:
        latents = inversed_latents
    else:
        set_seed(seed)
        latents = torch.randn_like(img_latents)
    
    target_img = img_latents.clone().to(torch.float32)

    # get the eta values for each steps in 
    eta_values = generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

    # handle guidance scale if need
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    print("TOTAL denosing timesteps:", len(timesteps))
    print("TIMESTEPS", timesteps)
          
    with pipeline.progress_bar(total=num_steps) as progress_bar:
        #for i, t in enumerate(timesteps):
        for i, t in enumerate(timesteps[-1:]):
            print("DENOSING TIMESTEP: ", t)
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # Editing text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance scale
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = pred_velocity.chunk(2)
                pred_velocity = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target image velocity
            t_curr = t / pipeline.scheduler.config.num_train_timesteps
            target_velocity = -(target_img - latents) / t_curr

            # interpolated velocity
            eta = eta_values[i]
            interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

            # denosing
            latents = pipeline.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]
            return latents, pred_velocity, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

            latents = latents.to(DTYPE)
            progress_bar.update()
    
    return latents

@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    gamma,
    DTYPE,
    prompt = "",
    num_steps=28,
    seed=42
):

    # using FlowMatchEulerDiscreteScheduler
    pipeline.scheduler.set_timesteps(num_steps, device=pipeline.device)

    # check if it has sigma avaible 
    if not hasattr(pipeline.scheduler, "sigmas"):
        raise Exception("Cannot find sigmas variable in scheduler. Please use FlowMatchEulerDiscreteScheduler to doing RF Inversion")
    
    # we get timestep directy from sigmas
    timesteps = pipeline.scheduler.sigmas
    timesteps = torch.flip(timesteps, dims=[0])

    #print("INVERSION_PAIR: ", timesteps[:2])
    print("TOTAL inversion timesteps:", len(timesteps))
    print("TIMESTEPS*1000", timesteps*1000)


    # Getting null-text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt( # null text
        prompt=prompt, 
        prompt_2=prompt,
        prompt_3=prompt,
    )

    # generate gaussain noise with seed
    set_seed(seed)
    target_noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)

    # # Image inversion with interpolated velocity field.  t goes from 0.0 to 1.0
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((latents.shape[0],), t_curr * 1000, dtype=latents.dtype, device=latents.device)
            print("INVERSION TIMESTEP: ", t_curr)

            # Null-text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=latents,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target noise velocity
            target_noise_velocity = (target_noise - latents) / (1.0 - t_curr)
            
            # interpolated velocity
            interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * pred_velocity
            
            # one step Euler, similar to pipeline.scheduler.step but in the forward to noise instead of denosing
            latents = latents + (t_prev - t_curr) * interpolated_velocity
            
            return latents, pred_velocity, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

            latents = latents.to(DTYPE)
            progress_bar.update()
            
    return latents

def main():
    # We want to know if the single step velocity got the perfect reconsturction. becacuse flow model should got the perfect one .

    # load pipeline 
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=DTYPE, safety_checker=None)
    pipe = pipe.to("cuda")

    # using FlowMatchEulerDiscreteScheduler
    pipe.scheduler.set_timesteps(NUM_STEP, device=pipe.device)

    # load image 
    img = Image.open("src/20241103/images/dog.jpg")

    train_transforms = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    img = train_transforms(img).unsqueeze(0).to(device).to(DTYPE)
    ori_latent = encode_imgs(img, pipe, DTYPE)

    # Inversion should give the same image 
    inversed_latent, inversion_velocity, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = interpolated_inversion(
        pipe, 
        ori_latent,
        gamma=0.0,
        DTYPE=DTYPE,
        prompt = "a photo of a dog",
        num_steps=NUM_STEP,
        seed=42
    )

    generate_latent, generation_velocity, _, _, _, _ = interpolated_denoise(
        pipe, 
        ori_latent.to(DTYPE),
        eta_base=0.0,
        eta_trend='constant',
        start_step=0,
        end_step=1,
        inversed_latents=inversed_latent.to(DTYPE),
        use_inversed_latents=True,
        guidance_scale=6.0,
        prompt="a photo of a dog",
        DTYPE=DTYPE,
        seed = 42,
        num_steps=NUM_STEP,
        prompt_embeds=prompt_embeds, 
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
    )
    print("is latent before and after 1 step close :", torch.isclose(generate_latent.float(), ori_latent.float(), 1e-4))
    print("is the velocity close :", torch.isclose(inversion_velocity.float(), -generation_velocity.float(), 1e-4))

    # Decode latents to images
    out = decode_imgs(generate_latent.to(DTYPE), pipe, output_type="pil")[0]
    out.save(f"single_step_{NUM_STEP}.png")

    

if __name__ == "__main__":
    main()