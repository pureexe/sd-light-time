import os
import torch
import json
import torchvision
import numpy as np
import ezexr
from constants import DATASET_ROOT_DIR



class EnvmapAffineTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        specific_file=None,
        normal_axis=True,
        dataset_multiplier=1,
        prompt_path=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.normal_axis = normal_axis
        self.specific_file = specific_file
        self.dataset_multiplier = dataset_multiplier
        self.files = self.get_image_files()

        if prompt_path is None:
            prompt_path = os.path.join(self.root_dir, "prompts.json")
        self.prompt = self.get_prompt_from_file(prompt_path)
            
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512),  # Resize the image to 512x512
        ])

    def get_prompt_from_file(self, file_path):
        with open(file_path) as f:
            prompt = json.load(f)
        return prompt
    
    def get_image(self, idx):
        image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(image_path) / 255.0
        image = image[:3]
        return image
    
    def get_ldr(self, idx):
        ldr_path = os.path.join(self.root_dir, "ev0", f"{self.files[idx]}.png")
        image = torchvision.io.read_image(ldr_path) / 255.0
        image = image[:3]
        image = torchvision.transforms.functional.resize(image, (256, 256))
        return image

    def get_normalized_hdr(self, idx):
        hdr_path = os.path.join(self.root_dir, "exr", f"{self.files[idx]}.exr")
        hdr = ezexr.imread(hdr_path).astype(np.float32)
        hdr = log_map_to_range(hdr)
        hdr = hdr.permute(2, 0, 1)
        hdr = hdr[:3] #only first 3 channel
        assert hdr.shape[1] > 3 and hdr.shape[2] > 3
        hdr = torchvision.transforms.functional.resize(hdr, (256, 256))
        return hdr

    def get_image_files(self):
        files = []
        for fname in sorted(os.listdir(os.path.join(self.root_dir, "exr"))):
            if fname.endswith(".exr"):
                fname = fname.replace(".exr","")
                files.append(fname)
        return files

    
    def __len__(self):
        return len(self.files) * self.dataset_multiplier 
    
    
    
    def __getitem__(self, idx):
        if self.dataset_multiplier > 1:
            word_idx = idx % self.dataset_multiplier
            word_name = f"{word_idx:05d}"
            idx = idx // self.dataset_multiplier
        else:
            word_name = self.files[idx]
            
        #pixel_values = self.transform(self.get_image(idx))
        pixel_values = torch.zeros(3, 512, 512).float()
        # if flip_type == 1:
        #     pixel_values = torchvision.transforms.functional.hflip(pixel_values)
        return {
            'name': self.files[idx],
            'pixel_values': pixel_values,
            'ldr_envmap': self.get_ldr(idx),
            'normalized_hdr_envmap': self.get_normalized_hdr(idx),
            'text': self.prompt[word_name],
            'word_name': word_name,
            'idx': idx,
        }
    
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