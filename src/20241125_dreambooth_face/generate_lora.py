import os 
import torch 
import json
from diffusers import StableDiffusion3Pipeline

SEED = 0

#EXP_NAME = "right/lr1e-4_rank4"
#EXT_PROMPT = ", with light coming from sks"

EXP_NAME = "right_321/lr1e-4_rank32"
EXT_PROMPT = ", with sunlight illuminate on the sks"

OUTPUT_DIR = "../../output/20241125_dreambooth_face/render"
PATH = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20241125_dreambooth_face/{EXP_NAME}"

def get_lastest_directory(path):
    checkpoint_dirs = os.listdir(PATH)
    checkpoint_dirs = [int(c.replace('checkpoint-','')) for c in checkpoint_dirs if c.startswith('checkpoint-')]
    lastest  = sorted(checkpoint_dirs)[-1]
    return os.path.join(path, f"checkpoint-{lastest}")

def get_prompts():
    with open('val_prompt100.json') as f:
        data = json.load(f)
    prompts = []
    for i in range(100):
        prompt = data[f"60000/600{i:02d}"]
        prompts.append(prompt +  EXT_PROMPT)
    return prompts

@torch.inference_mode()
def main():
    chkdir = get_lastest_directory(PATH)
    lora_path = os.path.join(chkdir, 'pytorch_lora_weights.safetensors')
    prompts = get_prompts()
    output_dir = os.path.join(OUTPUT_DIR, EXP_NAME)
    print("output_dir: ", output_dir)
    os.makedirs(output_dir,exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=1.0)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device=torch.device('cuda')).manual_seed(SEED)
    for idx in range(100):
        image = pipe(
            prompt=prompts[idx],
            generator=generator
        ).images[0]
        image.save(os.path.join(output_dir, f"{idx:02d}.png"))


if __name__ == "__main__":
    for exp in ["right_321/lr1e-4_rank32", "right_321/lr1e-4_rank4", "right_321/lr1e-4_rank8", "right_321/lr1e-4_rank16", "right_321/lr1e-4_rank32", "right_321/lr1e-4_rank32", "right_321/lr1e-5_rank4", "right_321/lr1e-5_rank8", "right_321/lr1e-5_rank16", "right_321/lr1e-5_rank32"]:
        EXP_NAME = exp
        PATH = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20241125_dreambooth_face/{EXP_NAME}"
        main()
