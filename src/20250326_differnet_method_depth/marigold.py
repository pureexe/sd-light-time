import diffusers
import torch

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250326_differnet_method_depth/input/dir_0_mip2.jpg")

normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals.prediction)
vis[0].save("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250326_differnet_method_depth/output/marigold.png")
