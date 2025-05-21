import os
import json
import torch 
from torchvision import transforms
import webdataset as wds
from tqdm.auto import tqdm
from PIL import Image
import io
import numpy as np
import itertools


TOTAL_FILES = 100000

class SizedWebDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset, length):
        self.base = base_dataset
        self._length = length

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return self._length

def decode_npz(npz_bytes):
    with io.BytesIO(npz_bytes) as buf:
        return np.load(buf)["arr_0"]

def main():
    
    INPUT_DIR = "/pure/f1/datasets/multi_illumination/least_square/v4_webdataset/train/train-{0000..024}.tar"
    
    # INPUT_DIR = []
    # for i in range(0, 25):
    #     INPUT_DIR.append(f"/pure/f1/datasets/multi_illumination/least_square/v4_webdataset/train/train-{i:04d}.tar")

    # INPUT_DIR = itertools.cycle(INPUT_DIR)  # infinite iterator in order


    image_transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    
    # dataset = (
    #     wds.WebDataset(INPUT_DIR, resampled=True)        
    #     .select(lambda sample: all(k in sample for k in ["jpg", "npz", "txt"]))
    #     .to_tuple("jpg", "npz", "txt", "__key__")
    #     .map(lambda data_tuple: {
    #         "pixel_values": image_transforms(Image.open(io.BytesIO(data_tuple[0])).convert("RGB")),
    #         "conditioning_pixel_values": decode_npz(data_tuple[1]),
    #         "prompts": data_tuple[2],
    #         "filename": data_tuple[3]
    #     })
    # )
    dataset = (
        wds.WebDataset(INPUT_DIR)        
        .select(lambda sample: all(k in sample for k in ["jpg", "npz", "txt"]))
        .to_tuple("__key__")
        .map(lambda data_tuple: {
            "filename": data_tuple[0]
        })
    )
    print(len(dataset))
    exit()
    #dataloader = dataloader = wds.WebLoader(dataset, batch_size=1, num_workers=8)

    #dataset = SizedWebDataset(dataset, TOTAL_FILES)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    filenames = []
    for idx, batch in enumerate(tqdm(dataloader)):
        if idx >= TOTAL_FILES:
            break
        filenames.append(batch["filename"][0])
    with open("filenames_no_resample.json", "w") as f:
        json.dump(filenames, f, indent=4)
    print(f"Saved {len(filenames)} filenames to filenames.json")    
    

if __name__ == "__main__":
    main()
    