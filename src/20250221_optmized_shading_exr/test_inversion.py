from diffusers import StableDiffusionPipeline
from InversionHelper import get_ddim_latents
import torch 
import skimage
import numpy as np 
import torchvision

SEED = 42 
MASTER_TYPE = torch.float16

def get_image():
    """
    Get image for use 
    @return image in range [-1,1] shape [b,c,h,w]
    """
    image = skimage.io.imread("/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_render_from_fitting_v2/14n_copyroom10/dir_4_mip2.png")
    image = skimage.img_as_float(image)
    image = skimage.transform.resize(image, (512,512))
    image = torch.tensor(image)
    image = image.permute(2,0,1)
    image = (image * 2) - 1.0
    image =  image[None]
    assert image.shape == (1,3,512,512)
    return image

@torch.inference_mode()
def main():
    NUM_INFERNECE_STEP = 500
    INTERRUPT_INDEX = None
    # load stable diffusion pipeline
    sd_path="runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        safety_checker=None,
        torch_dtype=MASTER_TYPE,
    )
    pipe = pipe.to('cuda')
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    device = pipe.device

    # get prompt 
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt = ["a photorealistic image"],
        device = device,
        num_images_per_prompt=1, 
        do_classifier_free_guidance=False,
        negative_prompt = [""] * 1
    )
    assert prompt_embeds.shape == (1,77,768)

    image = get_image().to(device).to(MASTER_TYPE)
    prompt_embeds = prompt_embeds
    

    ddim_latents, ddim_timesteps = get_ddim_latents(
        pipe = pipe,
        image=image,
        text_embbeding=prompt_embeds,
        num_inference_steps=NUM_INFERNECE_STEP,
        generator=torch.Generator(device=device).manual_seed(SEED),
        guidance_scale=1.0,
        interrupt_index = INTERRUPT_INDEX
    )

    pipe_args = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "output_type": "pt",
        "guidance_scale": 1.0,
        "return_dict": False,
        "num_inference_steps": NUM_INFERNECE_STEP,
        "generator": torch.Generator().manual_seed(SEED),
    }
    pipe_args["latents"] = ddim_latents[-1]
    pt_image, _ = pipe(**pipe_args)

    gt_image = (image + 1.0) / 2.0 #bump back to range[0-1]
    tb_image = [gt_image, pt_image]
    tb_image = torch.cat(tb_image, dim=0)

    tb_image = torchvision.utils.make_grid(tb_image, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
    torchvision.utils.save_image(tb_image, f"ddim_predicted.jpg")


if __name__ == "__main__":
    main()