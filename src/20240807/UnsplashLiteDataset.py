import os
import torch
import json
import torchvision
import numpy as np
import ezexr
from constants import DATASET_ROOT_DIR

ACCEPT_EXTENSION = ['jpg', 'png', 'jpeg', 'exr']
LDR_DIR = "env_ldr"
NORM_DIR = "env_norm"
IMAGE_DIR = "images"
IS_DEBUG = False

class UnsplashLiteDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        dataset_multiplier=1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_multiplier = dataset_multiplier
        self.files= self.get_image_files()
        self.prompt = self.get_prompt_from_file("prompts.json") 
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])

    def get_prompt_from_file(self, filename):
        with open(os.path.join(self.root_dir, filename)) as f:
            prompt = json.load(f)
        return prompt
    
    def get_image(self, idx: int):
        for ext in ACCEPT_EXTENSION:
            image_path = os.path.join(self.root_dir,  IMAGE_DIR, f"{self.files[idx]}.{ext}")
            if os.path.exists(image_path):
                image = torchvision.io.read_image(image_path) / 255.0
                image = image[:3]
                image = torchvision.transforms.functional.resize(image, (512, 512),  antialias=True)
                assert image.shape[1] == 512 and image.shape[2] == 512, "Only support 512x512 image"
                return image
        raise FileNotFoundError(f"File not found for {self.files[idx]}")
    
    def get_env_ldr(self, idx: int):
        for ext in ACCEPT_EXTENSION:
            ldr_path = os.path.join(self.root_dir, LDR_DIR, f"{self.files[idx]}.{ext}")
            if os.path.exists(ldr_path):
                image = torchvision.io.read_image(ldr_path) / 255.0
                image = torchvision.transforms.functional.resize(image, (256, 256),  antialias=True)
                image = image[:3]
                assert image.shape[1] == 256 and image.shape[2] == 256, "Only support 256x256 image"
                return image
        raise FileNotFoundError(f"File not found for {self.files[idx]}")

    def get_env_norm(self, idx: int):
        for ext in ACCEPT_EXTENSION:
            hdr_path = os.path.join(self.root_dir, NORM_DIR,  f"{self.files[idx]}.{ext}")
            if os.path.exists(hdr_path):
                if ext == "exr":
                    hdr = ezexr.imread(hdr_path)
                    hdr = log_map_to_range(hdr)
                else:
                    hdr = torchvision.io.read_image(hdr_path) / 255.0
                hdr = hdr.permute(2, 0, 1)
                hdr = torchvision.transforms.functional.resize(hdr, (256, 256),  antialias=True)
                hdr = torch.clamp(hdr, 0.0, 1.0)
                hdr = hdr[:3]
                assert hdr.shape[1] == 256 and hdr.shape[2] == 256, "Only support 256x256 image"
                return hdr
        raise FileNotFoundError(f"File not found for {self.files[idx]}")


    def get_image_files(self):
        files = []
        for fname in sorted(os.listdir(os.path.join(self.root_dir, IMAGE_DIR))):           
            ext = fname.split('.')[-1]
            if ext in ACCEPT_EXTENSION:
                to_replace = "."+ext
                fname = fname.replace(to_replace,"")
                files.append(fname)
        return files
    
    def __len__(self):
        return len(self.files) * self.dataset_multiplier 
    
    def __getitem__(self, idx):
        # support dataset_multiplier
        idx = idx % len(self.files)
        try:
            word_name = self.files[idx]
            pixel_values = self.transform(self.get_image(idx))
            return {
                    'name': self.files[idx],
                    'pixel_values': pixel_values,
                    'ldr_envmap': self.get_env_ldr(idx),
                    'normalized_hdr_envmap': self.get_env_norm(idx),
                    'text': self.prompt[word_name],
                    'word_name': word_name,
                    'idx': idx,
                }
        except Exception as e:
            if IS_DEBUG:
                raise e
            print(f"\nDataset retrival error at idx: {idx}\n")
            return self.__getitem__(idx+1)
    
def log_map_to_range(arr):
    """
    Maps an array containing values from 0 to positive infinity to [0, 1]
    using logarithmic scaling with the maximum value as 1, using PyTorch.

    Args:
        arr: A PyTorch tensor of floats.

    Returns:
        A PyTorch tensor of floats, scaled to the range [0, 1].
    """

    # Convert to PyTorch tensor if needed
    if not torch.is_tensor(arr):
        arr = torch.tensor(arr)
    m_arr = arr.clone() 

    # Handle potential zeros (replace with a small positive value)
    eps = np.finfo(float).eps
    arr = torch.clamp(arr, min=eps)
    
    # Find the maximum value (avoiding NaNs)
    max_value = torch.max(arr)

    if np.abs(max_value-1.0) < 1e-6:
        # No scaling needed
        scaled_arr = arr
    else:
        # Apply logarithmic scaling (base-10 for clarity)
        scaled_arr = torch.log10(arr) / torch.log10(max_value)

    # Clip to ensure values are within [0, 1] (optional)
    scaled_arr = torch.clamp(scaled_arr, min=0, max=1)

    return scaled_arr