import torch
import os
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from torchvision import transforms
from sd3RFinversion import interpolated_inversion, interpolated_denoise, encode_imgs, decode_imgs

DTYPE = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEP = 999

def main():
    # load pipeline 
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=DTYPE, safety_checker=None)
    pipe = pipe.to("cuda")

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
    img_latent = encode_imgs(img, pipe, DTYPE)

    # Inversion should give the same image 
    inversed_latent = interpolated_inversion(
        pipe, 
        img_latent,
        gamma=0.0,
        DTYPE=DTYPE,
        prompt = "a photo of a dog",
        num_steps=NUM_STEP,
        seed=42
    )

    img_latents = interpolated_denoise(
        pipe, 
        img_latent,
        eta_base=0.0,
        eta_trend='constant',
        start_step=0,
        end_step=1,
        inversed_latents=inversed_latent,
        use_inversed_latents=True,
        guidance_scale=6.0,
        prompt="a photo of a dog",
        DTYPE=DTYPE,
        seed = 42,
        num_steps=NUM_STEP
    )

    # Decode latents to images
    out = decode_imgs(img_latents, pipe, output_type="pil")[0]
    out.save(f"dog_inverted_step{NUM_STEP}.png")



if __name__ == "__main__":
    main()