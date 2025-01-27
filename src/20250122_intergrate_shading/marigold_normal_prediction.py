import diffusers
import torch

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals.prediction)
vis[0].save("einstein_normals.png")
