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
IS_DEBUG = True

class UnsplashLiteDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        dataset_multiplier=1,
        specific_prompt=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_multiplier = dataset_multiplier
        self.files= self.get_image_files()
        if 'split' in kwargs:
            if type(kwargs['split']) == list:
                self.files = kwargs['split']
            else:
                self.files = self.files[kwargs['split']]
        self.prompt = self.get_prompt_from_file("prompts.json") 
        if specific_prompt is not None:
            self.specific_prompt = specific_prompt
            if type(specific_prompt) == list:
                self.dataset_multiplier = self.dataset_multiplier * len(specific_prompt)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])


    def get_prompt_from_file(self, filename):
        with open(os.path.join(self.root_dir, filename)) as f:
            prompt = json.load(f)
        return prompt
    
    def get_image(self, idx: int, directory:str, height =512, width=512):
        for ext in ACCEPT_EXTENSION:
            image_path = os.path.join(self.root_dir,  directory, f"{self.files[idx]}.{ext}")
            if os.path.exists(image_path):
                image = torchvision.io.read_image(image_path) / 255.0
                image = image[:3]
                # if image is one channel, repeat it to 3 channels
                if image.shape[0] == 1:
                    image = torch.cat([image, image, image], dim=0)
                assert image.shape[1] == height and image.shape[2] == width, "Only support 512x512 image"
                return image
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
            name = self.files[idx]
            word_name = self.files[idx]
            pixel_values = self.transform(self.get_image(idx,"images", 512, 512))
            ldr_envmap = self.get_image(idx,"env_ldr", 256, 256)
            under_envmap = self.get_image(idx,"env_under", 256, 256)
            try:
                control_depth = self.transform(self.get_image(idx,"control_depth", 512, 512))
            except:
                control_depth = torch.zeros(3, 512, 512)
            
            try:
                chromeball = self.get_image(idx,"chromeball", 512, 512)
            except:
                chromeball = torch.zeros(3, 512, 512)

            if self.specific_prompt is not None:
                if type(self.specific_prompt) == list:
                    prompt_id = idx // len(self.files)
                    prompt = self.specific_prompt[prompt_id]
                    word_name = f"{word_name}_{prompt_id}"
                else:
                    prompt = self.specific_prompt
            else:
                prompt = self.prompt[word_name]
            return {
                    'name': name,
                    'source_image': pixel_values,
                    'control_depth': control_depth,
                    'chromeball_image': chromeball,
                    'ldr_envmap': ldr_envmap,
                    'norm_envmap': under_envmap,
                    'text': prompt,
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