import torch 
from AffineControl import AffineControl
from diffusers import StableDiffusionPipeline
from diffusers import ControlNetModel
from ball_helper import pipeline2controlnetinpaint
from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline, AutoencoderKL, DDIMScheduler 
from pipeline import PureIFPipeline
from transformers import T5EncoderModel
 
MASTER_TYPE = torch.float16
 
class AffineDepth(AffineControl):
   
    def setup_sd(self):
        super().setup_sd(sd_path="runwayml/stable-diffusion-v1-5", controlnet_path="lllyasviel/sd-controlnet-depth")

    def get_control_image(self, batch):
        assert torch.any(torch.abs(batch['control_depth'] - (-1)) > 1e-3)
        return batch['control_depth']
    
class AffineNormal(AffineControl):
   
    def setup_sd(self):
        super().setup_sd(sd_path="runwayml/stable-diffusion-v1-5", controlnet_path="lllyasviel/sd-controlnet-depth")

    def get_control_image(self, batch):
        assert torch.all(batch['control_normal'] >= 0.0) and torch.all(batch['control_normal'] <= 1.0)
        return batch['control_normal']
    
class AffineNormalBae(AffineControl):
   
    def setup_sd(self):
        super().setup_sd(sd_path="runwayml/stable-diffusion-v1-5", controlnet_path="lllyasviel/control_v11p_sd15_normalbae")

    def get_control_image(self, batch):
        assert torch.all(batch['control_normal_bae'] >= 0.0) and torch.all(batch['control_normal_bae'] <= 1.0)
        return batch['control_normal_bae']

class AffineDepthNormal(AffineControl):

    def setup_sd(self):
        controlnet_depth_path = "lllyasviel/sd-controlnet-depth"
        controlnet_normal_path = "lllyasviel/sd-controlnet-normal"
        super().setup_sd(
            sd_path="runwayml/stable-diffusion-v1-5",
            controlnet_path=[controlnet_depth_path, controlnet_normal_path]
        )

    def get_control_image(self, batch):
        assert torch.all(batch['control_depth'] >= 0) and torch.all(batch['control_depth'] <= 1)
        assert torch.all(batch['control_normal'] >= 0) and torch.all(batch['control_normal'] <= 1)
        return [batch['control_depth'], batch['control_normal']]

class AffineDepthNormalBae(AffineControl):

    def setup_sd(self):
        controlnet_depth_path = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet_normal_path = "lllyasviel/control_v11p_sd15_normalbae"
        super().setup_sd(
            sd_path="runwayml/stable-diffusion-v1-5",
            controlnet_path=[controlnet_depth_path, controlnet_normal_path]
        )

    def get_control_image(self, batch):
        assert torch.all(batch['control_depth'] >= 0) and torch.all(batch['control_depth'] <= 1)
        assert torch.all(batch['control_normal_bae'] >= 0) and torch.all(batch['control_normal_bae'] <= 1)
        return [batch['control_depth'], batch['control_normal_bae']]
    
class AffineNoControl(AffineControl):
   
    def setup_sd(self, sd_path="runwayml/stable-diffusion-v1-5"):
        # load pipeline
        self.pipe =  StableDiffusionPipeline.from_pretrained(
            sd_path,
            safety_checker=None,
            torch_dtype=MASTER_TYPE
        )
        
        # load unet from pretrain 
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        # load pipe_chromeball for validation 
        #controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=MASTER_TYPE)
        #self.pipe_chromeball = pipeline2controlnetinpaint(self.pipe, controlnet=controlnet_depth).to('cuda')
        self.pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "latent_model_input", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "cond_scale", "guess_mode", "image"]


# class AffineDeepFloyd(AffineControl):
   
#     def setup_sd(self):
#         USE_UPSCALE = False
#         USE_TEXT8BIT = True 

#         if MASTER_TYPE == torch.float16:
#             self.pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=MASTER_TYPE, watermarker=None).to('cuda')
#             #self.pipe.enable_model_cpu_offload()
#             self.pipe.watermarker = None
#             if USE_UPSCALE:
#                 self.pipe_upscale =  IFSuperResolutionPipeline.from_pretrained(
#                     "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=MASTER_TYPE, watermarker=None
#                 ).to('cuda')
#                 self.pipe_upscale.watermarker = None
#                 self.pipe_upscale.enable_model_cpu_offload()
#         else:
#             self.pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", text_encoder=text_encoder).to('cuda')
#             self.pipe.watermarker = None    
#             if USE_UPSCALE:
#                 self.pipe_upscale = IFSuperResolutionPipeline.from_pretrained(
#                     "DeepFloyd/IF-II-M-v1.0", text_encoder=None, watermarker=None
#                 ).to('cuda')
#                 self.pipe_upscale.watermarker = None
#                 self.pipe_upscale.enable_model_cpu_offload()


#         # disable pipe grad
#         self.pipe.unet.requires_grad_(False)
#         self.pipe.text_encoder.requires_grad_(False)

#         if USE_UPSCALE:
#             # disable pipe_upscale grad
#             self.pipe_upscale.unet.requires_grad_(False)

#         # load vae for encoder 
#         self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=MASTER_TYPE).to('cuda')
#         self.vae.requires_grad_(False)

#         # change scheduler
#         scheduler_config = {k: v for k, v in self.pipe.scheduler.config.items() if k != "variance_type"}
#         scheduler_config["variance_type"] = "fixed_small"
#         self.pipe.scheduler = DDIMScheduler.from_config(scheduler_config)


class AffineDeepFloyd(AffineControl):
   
    def setup_sd(self):
        USE_UPSCALE = False
        USE_TEXT8BIT = True 

        if MASTER_TYPE == torch.float16:
            self.pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=MASTER_TYPE, watermarker=None).to('cuda')
            #self.pipe.enable_model_cpu_offload()
            self.pipe.watermarker = None
            if USE_UPSCALE:
                self.pipe_upscale =  IFSuperResolutionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=MASTER_TYPE, watermarker=None
                ).to('cuda')
                self.pipe_upscale.watermarker = None
                self.pipe_upscale.enable_model_cpu_offload()
        else:
            self.pipe = PureIFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", text_encoder=text_encoder).to('cuda')
            self.pipe.watermarker = None    
            if USE_UPSCALE:
                self.pipe_upscale = IFSuperResolutionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-M-v1.0", text_encoder=None, watermarker=None
                ).to('cuda')
                self.pipe_upscale.watermarker = None
                self.pipe_upscale.enable_model_cpu_offload()


        # disable pipe grad
        self.pipe.unet.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

        if USE_UPSCALE:
            # disable pipe_upscale grad
            self.pipe_upscale.unet.requires_grad_(False)

        # load vae for encoder 
        vae_varience = "fp16" if MASTER_TYPE == torch.float16 else  None
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=MASTER_TYPE, variant=vae_varience).to('cuda')
        self.vae.requires_grad_(False)

        # change scheduler
        scheduler_config = {k: v for k, v in self.pipe.scheduler.config.items() if k != "variance_type"}
        scheduler_config["variance_type"] = "fixed_small"
        self.pipe.scheduler = DDIMScheduler.from_config(scheduler_config)
