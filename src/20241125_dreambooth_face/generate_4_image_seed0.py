import torch 
from diffusers import StableDiffusion3Pipeline

SEED = 0

@torch.inference_mode()
def main():
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    generator = torch.Generator(device=torch.device('cuda')).manual_seed(SEED)
    for idx in range(4):
        image = pipe(
            prompt="photo of a boy wearing a red hat, with light coming from sks",
            generator=generator
        ).images[0]
        image.save(f"{idx:02d}.png")


if __name__ == "__main__":
    main()
