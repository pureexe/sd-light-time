import os 
import json 
import numpy as np 
from PIL import Image
import torch 
import ezexr
import torchvision
from skimage.transform import resize
from tonemapper import TonemapHDR

class DiffusionRendererEnvmapDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/train"):
        self.root_dir = root_dir
        self.transform = {
            'image': torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
        }

        self.prompt = self.get_prompt_from_file('prompts.json', input_dir=self.root_dir) 
        self.files = list(self.prompt.keys())

    def get_prompt_from_file(self, filename, input_dir = None):
        input_dir = self.root_dir if input_dir is None else input_dir
        with open(os.path.join(input_dir, filename)) as f:
            prompt = json.load(f)
        return prompt
    
    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        if image.mode != 'RGB':
            raise ValueError(f"Image at {path} is not in RGB format.")
        if self.transform and 'image' in self.transform:
            image = self.transform['image'](image)
        return image
    
    def get_image(self, filename):
        return self.read_image(os.path.join(self.root_dir, 'images', filename+'.jpg')) 
    
    def get_albedo(self, filename):
        return self.read_image(os.path.join(self.root_dir, 'albedo', filename+'.png'))

    def get_depth(self, filename):
        depth_map = np.load(os.path.join(self.root_dir, 'depth', filename+'.npy'))

        #  find percentile 99 of the depth map
        percentile_99 = np.percentile(depth_map, 99)

        # clip depth map to 99th percentile
        depth_map = np.clip(depth_map, 0, percentile_99)

        inverted_depth_map = np.max(depth_map) - depth_map
        normalized_depth_map = inverted_depth_map / np.max(inverted_depth_map) # scale [0,1]

        # change to scale [-1,1]
        normalized_depth_map = normalized_depth_map * 2 - 1
        normalized_depth_map = normalized_depth_map[...,None]
        # repeat the depth map to have 3 channels
        normalized_depth_map = np.repeat(normalized_depth_map, 3, axis=-1)  # Shape: (height, width, 3)
        return normalized_depth_map  # Shape: (3, height, width)

    def get_normal(self, filename):
        normal = np.load(os.path.join(self.root_dir, 'normal', filename+'.npz'))
        # get first key from the normal 
        key = list(normal.keys())[0]
        normal = normal[key]  # Shape: (height, width, 3)
        return normal.astype(np.float32)  # Convert to float32 for consistency


    def get_light_dir(self, height = 256, width = 512):
        # create  from [0, 2pi] and [-pi/2, pi/2] 
        theta = np.linspace(0, 2 * np.pi, width)
        phi = np.linspace(-np.pi / 2, np.pi / 2, height)
        theta, phi = np.meshgrid(theta, phi)
        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta) * np.cos(phi)
        z = np.sin(phi)
        light_dir = np.stack([x, y, z], axis=-1)  # Shape: (height, width, 3)
        return light_dir.astype(np.float32)  # Convert to float32 for consistency


    def get_environment_map(self, filename):
        image = ezexr.imread(os.path.join(self.root_dir, 'envmap', filename+'.exr'))
        image = np.clip(image, 0, np.inf) # Ensure no negative values
        # resize to 256x256 using numpy bilinear interpolation
        image = resize(image, (256, 256), order=1, anti_aliasing=True)  # Resize to 256x256

        return image 

    def get_ldr(self, hdr_image):
        # Convert HDR to LDR
        hdr2ldr = TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)
        ldr_image, _, _ = hdr2ldr(hdr_image)
        # scale to [-1,1]
        ldr_image = (ldr_image - 0.5) * 2  # Scale to [-1, 1]
        return ldr_image

    def get_log_hdr(self, hdr_image):
        # Convert HDR to Log HDR
        log_image = np.log(hdr_image+1) / np.max(hdr_image)  # Using log1p to avoid log(0)
        log_image = (log_image - 0.5) * 2
        return log_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        name = filename
        albedo = self.get_albedo(filename) 
        image = self.get_image(filename)
        envmap = self.get_environment_map(filename)
        depth = self.get_depth(filename)
        normal = self.get_normal(filename)
        light_ldr = self.get_ldr(envmap)
        light_log_hdr = self.get_log_hdr(envmap)
        light_dir = self.get_light_dir(height=envmap.shape[0], width=envmap.shape[1])
        prompt = self.prompt[filename]
        return {
            'name': name,
            'albedo': albedo,
            'depth': numpy_hwc_to_torch_chw(depth),
            'normal': numpy_hwc_to_torch_chw(normal),
            'light_ldr': numpy_hwc_to_torch_chw(light_ldr),
            'light_log_hdr': numpy_hwc_to_torch_chw(light_log_hdr), 
            'light_dir': numpy_hwc_to_torch_chw(light_dir),          
            'image': image,
            'prompt': prompt,
        }

def numpy_hwc_to_torch_chw(numpy_img):
    """
    Convert numpy image from HWC to CHW format.
    """
    return torch.from_numpy(numpy_img).permute(2, 0, 1).float()  # Convert to float tensor and permute dimensions
