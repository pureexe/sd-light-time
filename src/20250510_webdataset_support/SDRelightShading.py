import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from InversionHelper import get_ddim_latents

MASTER_TYPE = torch.float16

class SDRelightShading():
    """
    Relight by Inversion shading
    """
    
    def __init__(self, controlnet_path, sd_path="runwayml/stable-diffusion-v1-5", seed=42):
        self.seed = seed
        self.guidance_scale = 1.0
        self.ddim_guidance_scale = 1.0
        self.num_inversion_steps = 500
        self.setup_sd(sd_path=sd_path, controlnet_path=controlnet_path)

    def set_seed(self, seed):
        self.seed = seed

    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5", controlnet_path=None):
        # create controlnet from unet. Unet take 6 channel input with are 3 channel of background and other 3 channel of shading
        #controlnet = ControlNetModel.from_pretrained(controlnet_path) 
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=MASTER_TYPE,
            use_safetensors=True
        )

        # load pipeline
        self.pipe =  StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )

        self.regular_scheduler = self.pipe.scheduler

        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        is_save_memory = False
        if is_save_memory:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()    
        self.pipe.to('cuda')

    def get_noise_from_latents(self, latents, seed=None):
        """
        Obtain gaussain noise same shape as the latents
        """
        if seed is not None:
            # create random genration seed
            torch.manual_seed(seed)
            noise = torch.randn_like(latents, memory_format=torch.contiguous_format)
        else:
            noise = torch.randn_like(latents)
        return noise         
    
    def get_ddim_latents(
        self, 
        source_image,
        source_shading,
        prompt_embeds
    ):
        ddim_latents, ddim_timesteps = get_ddim_latents(
                pipe=self.pipe,
                image=source_image,
                text_embbeding=prompt_embeds,
                num_inference_steps=self.num_inversion_steps,
                generator=torch.Generator().manual_seed(self.seed),
                controlnet_image=source_shading,
                guidance_scale=self.ddim_guidance_scale,
                controlnet_conditioning_scale = 1.0
            )
        return ddim_latents[-1]
        

    def relight(
        self, 
        source_image, 
        target_shading, 
        prompt="", 
        source_shading = None, 
        latents = None
    ):
        """
        Relight image with target shading and source shading
        """
        # get prompts 
        is_apply_cfg = self.guidance_scale > 1
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt = prompt,
            device = self.pipe.device,
            num_images_per_prompt=1, 
            do_classifier_free_guidance=is_apply_cfg,
            negative_prompt = [""] * len(prompt)
        )
        
        if source_shading is not None and latents is None:
            latents = self.get_ddim_latents(source_image, source_shading, prompt_embeds)

        pipe_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "output_type": "pt",
            "guidance_scale": self.guidance_scale,
            "return_dict": False,
            "num_inference_steps": self.num_inversion_steps,
            "generator": torch.Generator().manual_seed(self.seed),
            "controlnet_conditioning_scale": 1.0,
            "image": target_shading
        }
        if latents is not None:
            pipe_args['latents'] = latents

        pt_image, _ = self.pipe(**pipe_args)
        return {
            'image': pt_image,
            'init_latent': latents # return initial latent to avoid recompute which take a lot of time
        }


