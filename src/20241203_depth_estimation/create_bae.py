import os
from PIL import Image
from transformers import pipeline
from tqdm.auto import tqdm
import warnings
import torch
import numpy as np
import cv2
from controlnet_aux import NormalBaeDetector
from multiprocessing import Pool
from functools import partial
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from controlnet_aux.processor import Processor
from controlnet_aux.util import HWC3, resize_image
from einops import rearrange




class NormalBaeDetectorPT(NormalBaeDetector):
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            #normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            #normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return normal

def main():
    image_dir = "images"
    mode = 'bae'
    if mode == 'bae':
        # Normal BAE
        output_dir = "control_normal_bae"
        preprocessor = NormalBaeDetectorPT.from_pretrained("lllyasviel/Annotators")
        preprocessor.to('cuda')
    elif mode == 'depth':
        # Normal BAE
        output_dir = "control_normal"
        preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        preprocessor.to('cuda')
    elif mode == 'normal_miads':
        # Normal
        output_dir = "control_normal"
        preprocessor = Processor('normal_midas')
        preprocessor.to('cuda')
    else:
        print("Please use valid model")
        exit()

    image = Image.open("/ist/ist-share/vision/relight/datasets/unsplash-lite/train/images_2/-3LtGq_RPcY.jpg").convert("RGB")
    processed_image = preprocessor(image, output_type="pt")
    processed_image = np.array(processed_image)
    np.save("normal_bae.npy", processed_image)

    # os.makedirs(output_dir, exist_ok=True)

    # for image_file in tqdm(sorted(os.listdir(image_dir))):
    #     image_path = os.path.join(image_dir, image_file)
    #     image = Image.open(image_path).convert("RGB")  # Convert to RGB

    #     processed_image = preprocessor(image)

    #     output_path = os.path.join(output_dir, image_file.replace(".jpg",".png"))
    #     processed_image.save(output_path)
    # print("Depth estimation completed!")


if __name__ == "__main__":
    main()