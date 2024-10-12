import os 
import torch 
import torchvision 
import json
import ezexr 
import numpy as np

ACCEPT_EXTENSION = ['jpg', 'png', 'jpeg', 'exr']
IMAGE_DIR = "images"
IS_DEBUG = True

class BaseRelightDataset(torch.utils.data.Dataset):

    def __init__(self, 
        root_dir="",
        dataset_multiplier=1,
        specific_prompt=None,
        is_fliplr=False,
        index_file=None,
        use_envmap=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_multiplier = dataset_multiplier
        self.flip_lr = is_fliplr
        self.use_envmap = use_envmap
        self.prompt = self.get_prompt_from_file("prompts.json") 
        self.specific_prompt = specific_prompt
        self.setup_transform()

    def setup_transform(self):
        self.transform = {}
        # image need to be resize to shape 512x512 and normalize to [-1, 1]
        self.transform['image'] = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])
        # envmap need to be resize to shape 256x256
        transform_envmap = [torchvision.transforms.Resize(256,  antialias=True)]
        if self.flip_lr:
            transform_envmap.append(torchvision.transforms.RandomHorizontalFlip(p=1.0))
        self.transform['envmap'] = torchvision.transforms.Compose(transform_envmap)
        
        # transform control image which only rezie to 512x512
        self.transform['control'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])


    def get_image(self, name:str, directory:str, height =512, width=512,root_dir=None):
        root_dir = self.root_dir if root_dir is None else root_dir
        for ext in ACCEPT_EXTENSION:
            image_path = os.path.join(root_dir,  directory, f"{name}.{ext}")
            if os.path.exists(image_path):
                if ext == 'exr':
                    ezexr.write(image_path, image)
                else:
                    image = torchvision.io.read_image(image_path) / 255.0
                image = image[:3]
                # if image is one channel, repeat it to 3 channels
                if image.shape[0] == 1:
                    image = torch.cat([image, image, image], dim=0)
                #assert image.shape[1] == height and image.shape[2] == width, "Only support 512x512 image"
                return image
        raise FileNotFoundError(f"File not found for {name}")
    
    def get_control_image(self, name:str, directory:str, height =512, width=512):
        try:
            control_image = self.get_image(name, directory, height, width)
        except:
            control_image = torch.zeros(3, 512, 512)
        return control_image

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
        return len(self.image_index) * self.dataset_multiplier 

    def get_prompt_from_file(self, filename, input_dir = None):
        input_dir = self.root_dir if input_dir is None else input_dir
        with open(os.path.join(input_dir, filename)) as f:
            prompt = json.load(f)
        return prompt

    def check_output(self, output):
        assert 'name' in output, "name is not in output"
        assert 'source_image' in output, "source_image is not in output"
        return True

    def get_shcoeffs(self, name):
        shcoeffs_path = os.path.join(self.root_dir, "shcoeffs", f"{name}.npy")
        if os.path.exists(shcoeffs_path):
            shcoeffs = torch.tensor(np.load(shcoeffs_path))
        else:
            shcoeffs = torch.zeros(3,9)
        return shcoeffs

    def get_item(self, idx, batch_idx):
        name = self.files[idx]
        word_name = self.files[idx]

        pixel_values = self.transform['image'](self.get_image(name,"images", 512, 512))
        control_depth = self.transform['control'](self.get_image(name,"control_depth", 512, 512))   
        control_normal = self.transform['control'](self.get_image(name,"control_normal", 512, 512))
        control_normal_bae = self.transform['control'](self.get_image(name,"control_normal_bae", 512, 512))
        
        shcoeffs = self.get_shcoeffs(name).flatten()

        if self.specific_prompt is not None:
            if type(self.specific_prompt) == list:
                prompt_id = batch_idx // len(self.files)
                prompt = self.specific_prompt[prompt_id]
                word_name = f"{word_name}_{prompt_id}"
            else:
                prompt = self.specific_prompt
        else:
            prompt = self.prompt[word_name]
        output = {
            'name': name,
            'source_image': pixel_values,
            'control_depth': control_depth,
            'control_normal': control_normal,
            'control_normal_bae': control_normal_bae,
            'sh_coeffs': shcoeffs,
            'text': prompt,
            'word_name': word_name,
            'idx': idx,
        }
        if self.use_envmap:
            output['ldr_envmap'] = ldr_envmap
            output['norm_envmap'] = under_envmap
            ldr_envmap = self.transform['envmap'](self.get_image(name,"env_ldr", 256, 256))
            under_envmap = self.transform['envmap'](self.get_image(name,"env_under", 256, 256))
        return output

    
    def __getitem__(self, batch_idx):
        # support dataset_multiplier
        idx = batch_idx % len(self.files)
        try:
            output = self.get_item(idx, batch_idx)
            if self.check_output(output):
                return output
            else:
                raise ValueError("Output is not valid")
        except Exception as e:
            if IS_DEBUG:
                raise e
            print(f"\nDataset retrival error at idx: {idx}\n")
            return self.__getitem__(idx+1)