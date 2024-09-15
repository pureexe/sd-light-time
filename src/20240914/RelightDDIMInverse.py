
import os 
import torch 
import torchvision
from tqdm.auto import tqdm

from constants import *
from diffusers import DDIMScheduler, DDIMInverseScheduler

from LightEmbedingBlock import set_light_direction
from DDIMInversion import DDIMInversion
from AffineControl import AffineControl
from ball_helper import inpaint_chromeball
import numpy as np

 ## Inversion

@torch.inference_mode() 
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=350,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

def create_ddim_inversion(base_class):
    
    class RelightDDIMInverse(base_class):
        def __init__(self, *args,  **kwargs) -> None:
            super().__init__()

            # DEFAULT VARIABLE 
            self.target_timestep = 1000
            self.num_inference_steps = 200 # may need to use many steps to get the best result

            self.nomral_scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')
            self.inverse_scheduler = DDIMInverseScheduler.from_config(self.pipe.scheduler.config, subfolder='scheduler')  

            self.pipe.scheduler = self.nomral_scheduler

            # create ddim inversion
            self.ddim_inversion = DDIMInversion(self.pipe)
            
            # dsiable chromeball
            #del self.pipe_chromeball

        def generate_tensorboard(self, batch, batch_idx, is_save_image=False):
            USE_LIGHT_DIRECTION_CONDITION = False
            USE_OWN_DDIM = False 
            USE_HFDOC_DDIM = True
            # Apply the source light direction
            if USE_LIGHT_DIRECTION_CONDITION:
                set_light_direction(
                    self.pipe.unet, 
                    self.get_light_features(
                        batch['source_ldr_envmap'],
                        batch['source_norm_envmap'],
                    ), 
                    is_apply_cfg=self.guidance_scale > 1
                )
            else:
                set_light_direction(
                    self.pipe.unet, 
                    None, 
                    #is_apply_cfg=self.guidance_scale > 1
                    is_apply_cfg=False
                )

            # we first find the z0_noise before doing inversion
            negative_embedding = None
            with torch.inference_mode():
                z0_noise = self.pipe.vae.encode(batch['source_image']).latent_dist.sample(generator=torch.Generator().manual_seed(self.seed)) * self.pipe.vae.config.scaling_factor
                text_embbeding = self.get_text_embeddings(batch['text'])
                if self.guidance_scale > 1:
                    negative_embedding = self.get_text_embeddings([''])
                    negative_embedding = negative_embedding.repeat(text_embbeding.shape[0], 1, 1)

            if USE_OWN_DDIM:
                ddim_args = {
                    'z': z0_noise,
                    'embedd': text_embbeding,
                    'negative_embedd': negative_embedding,   
                    'target_timestep': self.target_timestep,
                    'num_inference_steps': self.num_inference_steps,
                    'guidance_scale': self.guidance_scale,
                    'device': z0_noise.device,
                }
                if hasattr(self.pipe, 'controlnet'):
                    ddim_args['controlnet_cond'] = self.get_control_image(batch)    
                    ddim_args['cond_scale'] = self.condition_scale
                    # DDIM inverse to get an intial noise
                zt_noise, _ = self.ddim_inversion(**ddim_args)
            elif USE_HFDOC_DDIM:
                zt_noise = invert(
                    self.pipe,
                    z0_noise,
                    batch['text'][0],
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.target_timestep,
                    do_classifier_free_guidance=self.guidance_scale > 1,
                    negative_prompt="",
                    device=z0_noise.device,
                )[-1][None]
            else:

                self.pipe.scheduler = self.inverse_scheduler

                zt_noise, _ = self.pipe(
                    prompt_embeds=text_embbeding,
                    negative_prompt_embeds=negative_embedding, 
                    guidance_scale=self.guidance_scale,
                    latents=z0_noise,
                    output_type='latent',
                    return_dict=False,
                    num_inference_steps=self.num_inference_steps,
                    generator=torch.Generator().manual_seed(self.seed)
                )

                self.pipe.scheduler = self.nomral_scheduler

            if USE_LIGHT_DIRECTION_CONDITION:
                #Apply the target light direction            
                set_light_direction(
                    self.pipe.unet,
                    self.get_light_features(
                        batch['target_ldr_envmap'],
                        batch['target_norm_envmap']
                    ),
                    is_apply_cfg=self.guidance_scale > 1
                )
            

            pipe_args = {
                #"prompt": batch["text"],    
                #"negative_prompt": [""],
                "prompt_embeds": text_embbeding,
                "negative_prompt_embeds": negative_embedding,
                "latents": zt_noise,
                "output_type": "pt",
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "return_dict": False,
                "generator": torch.Generator().manual_seed(self.seed)
            }
            if hasattr(self.pipe, "controlnet"):
                pipe_args["image"] = self.get_control_image(batch)
            pt_image, _ = self.pipe(**pipe_args)
            gt_image = (batch["source_image"] + 1.0) / 2.0
            tb_image = [gt_image, pt_image]

            if hasattr(self.pipe, "controlnet"):
                ctrl_image = self.get_control_image(batch)
                if isinstance(ctrl_image, list):
                    tb_image += ctrl_image
                else:
                    tb_image.append(ctrl_image)

            if hasattr(self, "pipe_chromeball"):
                with torch.inference_mode():
                    # convert pt_image to pil_image
                    to_inpaint_img = torchvision.transforms.functional.to_pil_image(pt_image[0].cpu())                
                    inpainted_image = inpaint_chromeball(to_inpaint_img,self.pipe_chromeball)
                    inpainted_image = torchvision.transforms.functional.to_tensor(inpainted_image).to(pt_image.device)
                    tb_image.append(inpainted_image[None])

           
            images = torch.cat(tb_image, dim=0)
            image = torchvision.utils.make_grid(images, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
            
            self.logger.experiment.add_image(f'image/{batch["name"][0]}', image, self.global_step)
            if is_save_image:
                os.makedirs(f"{self.logger.log_dir}/with_groudtruth", exist_ok=True)
                torchvision.utils.save_image(image, f"{self.logger.log_dir}/with_groudtruth/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                if hasattr(self.pipe, "controlnet"):
                    os.makedirs(f"{self.logger.log_dir}/control_image", exist_ok=True)
                    if isinstance(ctrl_image, list):
                        for i, c in enumerate(ctrl_image):
                            torchvision.utils.save_image(c, f"{self.logger.log_dir}/control_image/{batch['name'][0]}_{batch['word_name'][0]}_{i}.jpg")
                    else:
                        torchvision.utils.save_image(ctrl_image, f"{self.logger.log_dir}/control_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                os.makedirs(f"{self.logger.log_dir}/crop_image", exist_ok=True)
                torchvision.utils.save_image(pt_image, f"{self.logger.log_dir}/crop_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the target_ldr_envmap
                os.makedirs(f"{self.logger.log_dir}/target_ldr_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['target_ldr_envmap'], f"{self.logger.log_dir}/target_ldr_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the target_norm_envmap
                os.makedirs(f"{self.logger.log_dir}/target_norm_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['target_norm_envmap'], f"{self.logger.log_dir}/target_norm_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the source_ldr_envmap
                os.makedirs(f"{self.logger.log_dir}/source_ldr_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['source_ldr_envmap'], f"{self.logger.log_dir}/source_ldr_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save the source_norm_envmap
                os.makedirs(f"{self.logger.log_dir}/source_norm_envmap", exist_ok=True)
                torchvision.utils.save_image(batch['source_norm_envmap'], f"{self.logger.log_dir}/source_norm_envmap/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                if hasattr(self, "pipe_chromeball"):
                    os.makedirs(f"{self.logger.log_dir}/inpainted_image", exist_ok=True)
                    torchvision.utils.save_image(inpainted_image, f"{self.logger.log_dir}/inpainted_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
                # save prompt
                os.makedirs(f"{self.logger.log_dir}/prompt", exist_ok=True) 
                with open(f"{self.logger.log_dir}/prompt/{batch['name'][0]}_{batch['word_name'][0]}.txt", 'w') as f:
                    f.write(batch['text'][0])
                # save the source_image
                os.makedirs(f"{self.logger.log_dir}/source_image", exist_ok=True)
                torchvision.utils.save_image(gt_image, f"{self.logger.log_dir}/source_image/{batch['name'][0]}_{batch['word_name'][0]}.jpg")
            if self.global_step == 0 and batch_idx == 0:
                self.logger.experiment.add_text('text', batch['text'][0], self.global_step)
                self.logger.experiment.add_text('learning_rate', str(self.learning_rate), self.global_step)
            

        @torch.inference_mode()
        def get_text_embeddings(self, text):
            if isinstance(text, str):
                text = [text]
            
            tokens = self.pipe.tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
                return_overflowing_tokens=True,
            ).input_ids.to(self.pipe.text_encoder.device)
            return self.pipe.text_encoder(tokens).last_hidden_state

    return RelightDDIMInverse
