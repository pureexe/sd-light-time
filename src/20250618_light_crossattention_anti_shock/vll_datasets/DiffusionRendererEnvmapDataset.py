import os 
import json 
import numpy as np 
from PIL import Image
import torch 
import ezexr
import torchvision
from skimage.transform import resize
from vll_datasets.diffusionrenderer_mapper import rgb2srgb, reinhard, hdr2log, envmap_vec

class DiffusionRendererEnvmapDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir="/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/train",
            index_file=None,
            components = ['albedo', 'normal', 'depth', 'light', 'image'],
            *args,
            **kwargs
        ):
        self.root_dir = root_dir
        self.components = components
        self.transform = {
            'image': torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
        }
        self.prompt = self.get_prompt_from_file('prompts.json', input_dir=self.root_dir) 

        # build index
        if index_file is not None:
            with open(index_file) as f:
                index = json.load(f)
            self.image_index = index['image_index']
            self.envmap_index = index['envmap_index']
        else:
            self.image_index = list(self.prompt.keys())
            self.envmap_index = []
            for filename in self.image_index:
                self.envmap_index.append([filename])

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


    def get_light_dir(self, height = 512, width = 512):
        env_resolution = (height, width)
        env_dir = envmap_vec(env_resolution) #[H,W,3]
        env_dir = env_dir.permute(2, 0, 1)  # Change to (3, H, W)
        return env_dir 

    def get_environment_map(self, filename):
        exr_path = os.path.join(self.root_dir, 'envmap', filename+'.exr')
        image = ezexr.imread(exr_path)
        image = np.clip(image, 0, np.inf) # Ensure no negative values

        # resize to 512x512 using numpy bilinear interpolation
        image = resize(image, (512, 512), order=1, anti_aliasing=True)  # Resize to 512x512

        # convert to torch tensor and change to [C, H, W]
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Change to (C, H, W)

        return image 

    def get_ldr(self, hdr_image):
        # Convert HDR to LDR
        ldr_image = rgb2srgb(reinhard(hdr_image).clamp(0, 1))
        # scale to [-1,1]
        ldr_image = (ldr_image - 0.5) * 2  # Scale to [-1, 1]
        return ldr_image

    def get_log_hdr(self, hdr_image):
        # Convert HDR to Log HDR
        log_image = rgb2srgb(hdr2log(hdr_image).clamp(0, 1))  # Convert to log space
        log_image = (log_image - 0.5) * 2 # Scale to [-1, 1]
        return log_image

    def __len__(self):
        return len(self.image_index)
    
    def get_light(self, filename):
        envmap = self.get_environment_map(filename)
        light_ldr = self.get_ldr(envmap)
        light_log_hdr = self.get_log_hdr(envmap)
        light_dir = self.get_light_dir(height=envmap.shape[1], width=envmap.shape[2])

        return {
            'light_ldr': light_ldr, 
            'light_log_hdr': light_log_hdr,
            'light_dir': light_dir
        }
    
    def get_item(self, idx):
        filename = self.image_index[idx]
        prompt = self.prompt[filename]
        output = {
            'name': filename,
            'text': prompt,
        }
        if 'albedo' in self.components:
            output['albedo'] = self.get_albedo(filename)
        if 'depth' in self.components:
            output['depth'] = numpy_hwc_to_torch_chw(self.get_depth(filename))
        if 'image'in self.components:
            output['image']  = self.get_image(filename)
        if 'light'in self.components:
            light = self.get_light(filename)
            output['light_ldr'] = light['light_ldr']
            output['light_log_hdr'] = light['light_log_hdr']
            output['light_dir'] = light['light_dir']
        if 'normal'in self.components:
            output['normal'] = numpy_hwc_to_torch_chw(self.get_normal(filename))
        return output

    def __getitem__(self, idx):
        return self.get_item(idx)

def numpy_hwc_to_torch_chw(numpy_img):
    """
    Convert numpy image from HWC to CHW format.
    """
    return torch.from_numpy(numpy_img).permute(2, 0, 1).float()  # Convert to float tensor and permute dimensions
