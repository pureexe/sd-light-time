import os
import torch
import json
import torchvision
import numpy as np
import ezexr
from constants import DATASET_ROOT_DIR



class EnvmapSelfAffineDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        split="train",
        specific_file=None,
        val_hold=100,
        val_fly=2,
        train_count=2000,
        normal_axis=True,
        dataset_multiplier=1,
        use_sh_light=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.normal_axis = normal_axis
        self.train_count = train_count
        self.val_hold = val_hold
        self.val_fly = val_fly
        self.split = split
        self.specific_file = specific_file
        self.dataset_multiplier = dataset_multiplier
        self.use_sh_light = use_sh_light
        self.files, self.subdirs = self.get_image_files()

        if self.use_sh_light and self.normal_axis:
            self.axis_low_end, self.axis_high_end = self.compute_normalize_bound(percentile=99.9)
            assert (np.abs(self.axis_high_end - self.axis_low_end)).sum() > 0.0
            
        self.files, self.subdirs = self.get_split_dataset(split)
        self.prompt = self.get_prompt_from_file("prompts.json") 
        

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512),  # Resize the image to 512x512
        ])

    def get_prompt_from_file(self, filename):
        with open(os.path.join(self.root_dir, filename)) as f:
            prompt = json.load(f)
        return prompt
    
    def get_image(self, idx):
        image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(image_path) / 255.0
        image = image[:3]
        return image
    
    def get_ldr(self, idx):
        ldr_path = os.path.join(self.root_dir, "ev0", self.subdirs[idx], f"{self.files[idx]}.jpg")
        if not os.path.exists(ldr_path):
            ldr_path = os.path.join(self.root_dir, "ev0", self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(ldr_path) / 255.0
        image = image[:3]
        image = torchvision.transforms.functional.resize(image, (256, 256))
        return image

    def get_normalized_hdr(self, idx):
        hdr_path = os.path.join(self.root_dir, "exr", self.subdirs[idx], f"{self.files[idx]}.jpg")
        if not os.path.exists(hdr_path):
            hdr_path = os.path.join(self.root_dir, "exr", self.subdirs[idx], f"{self.files[idx]}.exr")
            hdr = ezexr.imread(hdr_path)
            hdr = log_map_to_range(hdr)
            hdr = hdr.permute(2, 0, 1)
            hdr = hdr[:3] #only first 3 channel
        else:
            hdr = torchvision.io.read_image(hdr_path) / 255.0
            hdr = hdr[:3]
        
        hdr = torchvision.transforms.functional.resize(hdr, (256, 256))
        hdr = torch.clamp(hdr, 0.0, 1.0)
        assert hdr.shape[1] == 256 and hdr.shape[2] == 256, "Only support 256x256 image"
        # if torch.min(hdr) < 0.0:
        #     print(f"Negative value in HDR: {torch.min(hdr)} / {idx}")
        # if torch.max(hdr) > 1.0:
        #     print(f"Over 1.0 value in HDR: {torch.max(hdr)} / {idx}")
        # assert torch.min(hdr) >= 0.0 and torch.max(hdr) <= 1.0
        return hdr

    def get_image_files(self):
        files = []
        subdirs = []
        if self.specific_file is not None:
            with open(os.path.join(self.root_dir,  self.specific_file)) as f:
                data = json.load(f)
            for idx in range(len(data)):
                path = data[idx]
                subdir, fname = path.split("/")
                files.append(fname)
                subdirs.append(subdir)
        else:
            for subdir in sorted(os.listdir(os.path.join(self.root_dir, "images"))):
                for fname in sorted(os.listdir(os.path.join(self.root_dir, "images", subdir))):
                    if fname.endswith(".png"):
                        fname = fname.replace(".png","")
                        files.append(fname)
                        subdirs.append(subdir)
                    if fname.endswith(".jpg"):
                        fname = fname.replace(".jpg","")
                        files.append(fname)
                        subdirs.append(subdir)
        return files, subdirs

    def get_split_dataset(self, split):
        # if split is index slice 
        if type(split) != str:
            return self.files[split], self.subdirs[split]
        elif split == "train":
            return self.files[self.val_hold:self.val_hold+self.train_count], self.subdirs[self.val_hold:self.val_hold+self.train_count]
        elif split == "val":
            return self.files[:self.val_fly] + self.files[self.val_hold:self.val_hold+self.val_fly], self.subdirs[:self.val_fly] + self.subdirs[self.val_hold:self.val_hold+self.val_fly]
        elif ":" in split:
            start, end = map(int, split.split(":"))
            return self.files[start:end], self.subdirs[start:end]
        raise ValueError(f"split {split} not supported")
    
    def compute_normalize_bound(self, percentile=99.9):
        axis = []
        files, subdirs = self.get_split_dataset('train')
        axis_ids = [1,2,3]
        rows = {}
        for idx in range(len(files)):
            light = np.load(os.path.join(self.root_dir, "light", subdirs[idx], f"{files[idx]}_light.npy")) 
            light = self.convert_to_grayscale(light.transpose())
            for axis_id in axis_ids:
                if axis_id not in rows:
                    rows[axis_id] = []
                rows[axis_id].append(light[axis_id])
        low_ends = []
        high_ends = []
        for axis_id in axis_ids:
            axis = np.array(rows[axis_id])
            low_end = np.percentile(axis, 100-percentile)
            high_end = np.percentile(axis, percentile)
            low_ends.append(low_end)
            high_ends.append(high_end)
        return np.array(low_ends), np.array(high_ends)
        
    def __len__(self):
        return len(self.files) * self.dataset_multiplier 
    
    def convert_to_grayscale(self, v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        assert v.shape[0] == 3
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]

    def get_light_direction(self, idx):
        light = np.load(os.path.join(self.root_dir, "light", self.subdirs[idx], f"{self.files[idx]}_light.npy")) 
        light = self.convert_to_grayscale(light.transpose())
        assert len(light) == 9
        direction = light[1:4]
        if self.normal_axis:    
            direction = (direction - self.axis_low_end) / (self.axis_high_end - self.axis_low_end)
            direction = np.clip(direction, 0.0, 1.0)
            direction = direction * 2.0 - 1.0
        
        return direction 
    
    def __getitem__(self, idx):
        try:
            if self.dataset_multiplier > 1:
                word_idx = idx % self.dataset_multiplier
                #word_name = f"{word_idx:05d}"
                word_name = self.files[word_idx]
                idx = idx // self.dataset_multiplier
            else:
                word_name = self.files[idx]
                
            pixel_values = self.transform(self.get_image(idx))
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
        except Exception as e:
            print(f"\nDataset retrival error at idx: {idx}\n")
            raise e
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