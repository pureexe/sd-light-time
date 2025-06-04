# Tonemap first dataset to viewable format (Eg. .jpg)

import argparse 
import webdataset as wds
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from tonemapper import TonemapHDR
from tqdm.auto import tqdm
import os
import json

parser = argparse.ArgumentParser(description="Tonemap dataset to viewable format")
parser.add_argument(
    "--input_dir",
    type=str,
    default="/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v1/train/train-0000.tar",
    help="Path to the dataset to be tonemapped",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250518_dataset_inspection/real_image_lstsq_shading",
    help="Path to the output directory where the tonemapped images will be saved",
)
parser.add_argument(
    "--viz_tonemap_dir",
    type=str,
    default="viz_tone",
)

parser.add_argument(
    "--viz_max_dir",
    type=str,
    default="viz_max",
)

parser.add_argument(
    "--image_dir",
    type=str,
    default="images",
)

parser.add_argument(
    "--prompt_dir",
    type=str,
    default="prompts",
)

args = parser.parse_args()


class SizedWebDataset(torch.utils.data.IterableDataset):
        def __init__(self, base_dataset, length):
            self.base = base_dataset
            self._length = length

        def __iter__(self):
            return iter(self.base)

        def __len__(self):
            return self._length

def main():
    # Load the dataset
    
    viz_tonemap_dir = os.path.join(args.output_dir, args.viz_tonemap_dir)
    viz_max_dir = os.path.join(args.output_dir, args.viz_max_dir) 
    image_dir = os.path.join(args.output_dir, args.image_dir) 
    prompt_dir = os.path.join(args.output_dir, args.prompt_dir) 

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(viz_tonemap_dir, exist_ok=True)
    os.makedirs(viz_max_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)


    def decode_npz(npz_bytes):
        with io.BytesIO(npz_bytes) as buf:
            return np.load(buf)["arr_0"]

    tonemap = TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    
    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = (
        wds.WebDataset(args.input_dir, resampled=True)        
        .select(lambda sample: all(k in sample for k in ["jpg", "npz", "txt"]))
        .to_tuple("jpg", "npz", "txt", "__key__")
        .map(lambda data_tuple: {
            "pixel_values": image_transforms(Image.open(io.BytesIO(data_tuple[0])).convert("RGB")),
            "conditioning_pixel_values": decode_npz(data_tuple[1]),
            "prompts": data_tuple[2],
            "filename": data_tuple[3]
        })
    )

    dataset = SizedWebDataset(dataset, 1000)

    filenames = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, batch in enumerate(tqdm(dataloader)):
        if idx > 1000:
            break
        pixel_values = batch["pixel_values"][0]
        conditioning_pixel_values = batch["conditioning_pixel_values"][0]
        prompts = batch["prompts"][0]
        filename = batch["filename"][0]
        filenames.append(filename)
        pixel_values = pixel_values.permute(1, 2, 0).numpy()
        pixel_values = Image.fromarray((pixel_values * 255).astype(np.uint8))
        pixel_values.save(os.path.join(image_dir, filename + ".jpg"))
        conditioning_pixel_values = conditioning_pixel_values.numpy()
        
        ldr_image, _, _, = tonemap(conditioning_pixel_values)

        max_image = conditioning_pixel_values / np.max(conditioning_pixel_values)
        ldr_image = Image.fromarray((ldr_image * 255).astype(np.uint8))
        max_image = Image.fromarray((max_image * 255).astype(np.uint8))
        ldr_image.save(os.path.join(viz_tonemap_dir, filename + ".jpg"))
        max_image.save(os.path.join(viz_max_dir, filename + ".jpg"))
        with open(os.path.join(prompt_dir, filename + ".txt"), "w") as f:
            f.write(str(prompts))

    with open(os.path.join(args.output_dir, "filenames.json"), "w") as f:
        json.dump(filenames, f)

    # # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # # Iterate through the images in the input directory
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".exr"):
    #         # Load the image
    #         img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_UNCHANGED)

    #         # Tonemap the image
    #         tonemapped_img = cv2.createTonemap(2.2).process(img)

    #         # Save the tonemapped image to the output directory
    #         cv2.imwrite(os.path.join(output_dir, filename.replace(".exr", ".jpg")), tonemapped_img)
    #         # Save the tonemapped image to the viz_tonemap_dir



if __name__ == "__main__":
    main()