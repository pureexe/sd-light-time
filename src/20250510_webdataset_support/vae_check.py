import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from pathlib import Path

# Load Stable Diffusion 1.5 VAE
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
vae.eval().cuda()

# Image loading and preprocessing
image_path = "/pure/t1/datasets/laion-shading/v4/train/images/000000/000016.jpg"  # path to your input image
image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(512, interpolation=Image.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image_tensor = preprocess(image).unsqueeze(0).cuda()  # (1, 3, 512, 512)

# Encode with VAE
with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.sample() * 0.18215

# Decode with VAE
with torch.no_grad():
    decoded = vae.decode(latents / 0.18215).sample

# Postprocess and save
decoded = (decoded.clamp(-1, 1) + 1) / 2  # scale to [0,1]
to_pil = transforms.ToPILImage()
output_image = to_pil(decoded.squeeze().cpu())

output_path = Path("output/sofar.jpg")
output_image.save(output_path)
print(f"Saved decoded image to {output_path}")
