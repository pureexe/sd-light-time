import torch
import torchvision

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

try:
    import bitsandbytes as bnb
    USE_BITSANDBYTES = True
except:
    USE_BITSANDBYTES = False


from InversionHelper import get_ddim_latents, get_text_embeddings, get_null_embeddings, apply_null_embedding

from PIL import Image 

INPUT_IMAGE = "../../src/20240923/copyroom10.png"
MASTER_TYPE = torch.float32
PROMPT = "several pots of plants sit on a counter top"
NUM_INFERENCE_STEPS = 10
SEED = 42
GUIDANCE_SCALE = 7.0
NULL_STEP = 10
EXP_NAME = f"latentinput_simplify_{NUM_INFERENCE_STEPS}_3"
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def main():
    
    image = Image.open(INPUT_IMAGE)
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    # resize image to 512x512
    image = torchvision.transforms.Resize((512, 512))(image)
    # rescale image to [-1,1] from [0,1]
    image = image * 2 - 1

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=MASTER_TYPE,safety_checker=None).to(DEVICE)
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "latent_model_input", "timestep_cond", "added_cond_kwargs", "extra_step_kwargs", "noise_pred", "timesteps"]

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, subfolder='scheduler')

    # get text embedding
    text_embbeding = get_text_embeddings(pipe, PROMPT)
    negative_embedding = get_text_embeddings(pipe, '')

    # get DDIM inversion
    ddim_latents = get_ddim_latents(
        pipe, image,
        text_embbeding,
        NUM_INFERENCE_STEPS,
        torch.Generator().manual_seed(SEED)
    )
    null_embeddings, null_latents = get_null_embeddings(
        pipe,
        ddim_latents=ddim_latents,
        text_embbeding=text_embbeding,
        negative_embedding=negative_embedding,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        controlnet_image= None,
        num_null_optimization_steps=NULL_STEP,
        generator=torch.Generator().manual_seed(SEED)
    )
    pt_image = apply_null_embedding(
        pipe,
        ddim_latents[-1],
        null_embeddings,
        text_embbeding,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator().manual_seed(SEED),
        controlnet_image=None,
        null_latents=null_latents
    )
    torchvision.utils.save_image(pt_image, "output/nulltext.png")


if __name__ == "__main__":
    main()