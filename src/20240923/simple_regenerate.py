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
NUM_INFERENCE_STEPS = 300
GUIDANCE_SCALE = 7.0
NULL_STEP = 10
EXP_NAME = "300_4090_shift2"
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
    inverse_scheduler = DDIMInverseScheduler.from_config(nomral_scheduler.config, subfolder='scheduler')

    pipe.scheduler = nomral_scheduler

    # prepare inputs
    text_embbeding = get_text_embeddings(pipe, PROMPT)
    negative_embedding = get_text_embeddings(pipe, "")

    zt_noise = torch.load("zt_noise_300_4090.pt")
    #zt_noise = pipe.vae.encode(image.to(DEVICE).to(MASTER_TYPE)).latent_dist.sample(generator=torch.Generator().manual_seed(SEED)) * pipe.vae.config.scaling_factor

    null_embeddings = torch.load("null_embeddings_300_4090.pt")

    def callback_apply_nulltext(pipe, step_index, timestep, callback_kwargs):
    
        # skip the last step
        if step_index + 1 >= len(null_embeddings):
            return callback_kwargs
        _, text_embbeding = callback_kwargs['prompt_embeds'].chunk(2)
        # apply null text
        callback_kwargs['prompt_embeds'] = torch.cat([null_embeddings[step_index], text_embbeding])
        return callback_kwargs
    
    pipe_args = {
        "prompt_embeds": text_embbeding,
        "negative_prompt_embeds": negative_embedding,
        "latents": zt_noise.clone(),
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": torch.Generator().manual_seed(SEED),
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
        "callback_on_step_end": callback_apply_nulltext,
    }
    output_image = pipe(**pipe_args)['images'][0]
    output_image.save(f"output_copyroom10_regenerate_{EXP_NAME}.jpg")






if __name__ == "__main__":
    main()