import os
import torch
import json
import torchvision
import numpy as np
import ezexr
from constants import DATASET_ROOT_DIR

"""
Relight Envmap Dataset 
Expected the output to be image_source (no target, )
"""

class RelightEnvmapDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        prompt_file="prompts.json",
        index_file=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.files, self.subdirs = self.get_image_files()
        if index_file is None:
            # no index mean all index
            self.image_index = range(len(self.files))
            self.envmap_index = range(len(self.files))
        else:
            with open(index_file) as f:
                indexes = json.load(f)
                if not 'image_index' in indexes:
                    raise ValueError("image_index not found in index_file")
                if not 'envmap_index' in indexes:
                    raise ValueError("envmap_index not found in index_file")
                self.image_index = indexes['image_index']
                self.envmap_index = indexes['envmap_index']

            
        self.prompt = self.get_prompt_from_file(prompt_file)
        

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512),  # Resize the image to 512x512
        ])

    def get_prompt_from_file(self, filename):
        with open(os.path.join(self.root_dir, filename)) as f:
            prompt = json.load(f)
        return prompt
    
    def get_image(self, idx: int):
        """obtain the rgb image the dataset

        Args:
            idx (int): id in the dataset

        Returns:
            np.array: image in shape [3, 512, 512]
        """
        image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(image_path) / 255.0
        image = image[:3]

        assert len(image.shape) == 3 and image.shape[0] == 3
        return image
    
    def get_ldr(self, idx):
        """obtain Environment map (EV0) from the dataset

        Args:
            idx (int): index in the dataset

        Returns:
            np.array: image in shape [3, 256, 256] in range [0, 1]
        """
        ldr_path = os.path.join(self.root_dir, "ev0", self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(ldr_path) / 255.0
        image = image[:3]
        image = torchvision.transforms.functional.resize(image, (256, 256))

        assert len(image.shape) == 3 and image.shape[0] == 3
        assert image.shape[1] == 256 and image.shape[2] == 256
        assert torch.max(image) <= 1.0 and torch.min(image) >= 0.0
        return image

    def get_normalized_hdr(self, idx: int):
        """obtain Normalized HDR from the dataset

        Args:
            idx (int): index in the dataset  

        Returns:
            np.array: image in shape [3, 256, 256] in range [0, 1]
        """
        hdr_path = os.path.join(self.root_dir, "exr", self.subdirs[idx], f"{self.files[idx]}.exr")
        hdr = ezexr.imread(hdr_path)
        hdr = log_map_to_range(hdr)
        hdr = hdr.permute(2, 0, 1)
        hdr = hdr[:3] #only first 3 channel

        image = torchvision.transforms.functional.resize(hdr, (256, 256))

        assert len(image.shape) == 3 and image.shape[0] == 3
        assert image.shape[1] == 256 and image.shape[2] == 256
        assert torch.max(image) <= 1.0 and torch.min(image) >= 0.0

        return image

    def get_image_files(self):
        """get avalible image file and directory

        Returns:
            list: list of filenames
            list: list of subdirectories
        """
        files = []
        subdirs = []
        for subdir in sorted(os.listdir(os.path.join(self.root_dir, "images"))):
            for fname in sorted(os.listdir(os.path.join(self.root_dir, "images", subdir))):
                if fname.endswith(".png"):
                    fname = fname.replace(".png","")
                    files.append(fname)
                    subdirs.append(subdir)
        assert len(files) == len(subdirs)
        assert len(files) > 0
        return files, subdirs
        
    def __len__(self):
        return len(self.files)
    
    def convert_to_grayscale(self, v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        assert v.shape[0] == 3
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]

    
    def __getitem__(self, idx):

        image_index = self.image_index[idx]
        envmap_index = self.envmap_index[idx]        
        
        env_name = self.files[envmap_index]
        image_name = self.files[image_index]

        pixel_values = self.transform(self.get_image(image_index))
        
        
        return {
            'name': image_name,
            'pixel_values': pixel_values,
            'source_envmap_ldr': self.get_ldr(image_index),
            'source_envmap_hdr': self.get_normalized_hdr(image_index),
            'target_envmap_ldr': self.get_ldr(envmap_index),
            'target_envmap_hdr': self.get_normalized_hdr(envmap_index),
            'text': self.prompt[image_name],
            'env_name': env_name,
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