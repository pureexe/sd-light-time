import os
import torch
import json
import torchvision
import numpy as np
from constants import DATASET_ROOT_DIR



class FaceThreeAxisDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
        root_dir=DATASET_ROOT_DIR,
        split="train",
        specific_file=None,
        val_hold=100,
        val_fly=2,
        train_count=2000,
        normal_axis=True,
        dataset_multiplier=1,
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
        self.files, self.subdirs = self.get_image_files()

        if self.normal_axis:
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
        image_path = os.path.join(self.root_dir, "images",  self.subdirs[idx], f"{self.files[idx]}.png")
        image = torchvision.io.read_image(image_path) / 255.0
        return image

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
        # flip_type = idx % 2
        # idx = idx // 2
        if self.dataset_multiplier > 1:
            idx = idx // self.dataset_multiplier
        pixel_values = self.transform(self.get_image(idx))
        # if flip_type == 1:
        #     pixel_values = torchvision.transforms.functional.hflip(pixel_values)
        return {
            'name': self.files[idx],
            'pixel_values': pixel_values,
            'light': self.get_light_direction(idx),
            'text': self.prompt[self.files[idx]]
        }