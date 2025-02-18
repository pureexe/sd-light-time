import os 
import torch 
import torchvision 
import json
import ezexr 
import numpy as np
import random
from skimage.color import rgb2lab
import re 

ACCEPT_EXTENSION = ['jpg', 'png', 'jpeg', 'exr']
IMAGE_DIR = "images"
IS_DEBUG = True

class DiffusionFaceRelightDataset(torch.utils.data.Dataset):

    def __init__(self, 
        root_dir="",
        dataset_multiplier=1,
        specific_prompt=None,
        prompt_file="prompts.json",
        index_file=None,
        use_shcoeff2=False,
        random_mask_background_ratio=None,
        feature_types = ['shape', 'cam', 'faceemb', 'shadow', 'light'],
        light_dimension = 27, #light dim
        shadow_index = -1,
        use_ab_background=False,
        backgrounds_dir = "backgrounds", 
        shadings_dir = "shadings",
        images_dir = "images",
        use_background_jitter=False,
        use_false_shading=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        print("FEATURE_TYPE: ", feature_types)
        print("IMAGE_DIR: ", images_dir)
        self.root_dir = root_dir
        self.feature_types = feature_types
        self.dataset_multiplier = dataset_multiplier
        self.prompt = self.get_prompt_from_file(prompt_file) 
        self.specific_prompt = specific_prompt
        self.use_shcoeff2 = use_shcoeff2
        self.light_dimension = light_dimension
        self.random_mask_background_ratio = random_mask_background_ratio
        self.use_ab_background = use_ab_background
        self.use_false_shading = use_false_shading
        if 'shadow' in feature_types and 'light' in feature_types and feature_types[-1] == 'light' and shadow_index == -1:
            shadow_index = -28
        self.shadow_index = shadow_index # swap light shadow
        self.backgrounds_dir = backgrounds_dir #control_shading_from_ldr27coeff
        self.shadings_dir = shadings_dir
        self.use_background_jitter = use_background_jitter
        self.images_dir = images_dir

        self.setup_transform()
        self.setup_diffusion_face()

        # setup image index
        if index_file != "" and index_file != None:
            self.index_file = index_file
            self.build_index()
        else:
            self.index_file = self.get_avaliable_images()
            self.image_index = self.index_file
        self.files = self.image_index

    def build_index(self):
        if isinstance(self.index_file, dict):
            self.image_index = self.index_file['image_index']
            self.envmap_index = self.index_file['envmap_index']
        # check if index_file exists
        elif os.path.exists(self.index_file):
            with open(self.index_file) as f:
                index = json.load(f)
            self.image_index = index['image_index']
            self.envmap_index = index['envmap_index']
        else:
            raise ValueError("index_file should be a dictionary or a path to a file")
                
        assert len(self.image_index) == len(self.envmap_index), "image_index and envmap_index should have the same length"

    def get_avaliable_images(self, directory_path = None, accept_extensions=ACCEPT_EXTENSION):
        """
        Recursively get the list of files from a directory that match the accepted extensions.
        The file list is sorted by directory and then by filename.

        Args:
        - directory_path (str): Path to the directory to search.
        - accept_extensions (tuple or list): Tuple or list of acceptable file extensions (e.g., ('.jpg', '.png')).

        Returns:
        - List of sorted file paths that match the accepted extensions.
        """

        # Set default directory path
        if directory_path is None:
            directory_path = os.path.join(self.root_dir, self.images_dir)

        matched_files = []

        # Walk through directory and subdirectories
        for root, directory, files in os.walk(directory_path):
            # Filter files by extension
            for file in sorted(files):
                for accept_extension in accept_extensions:
                    if file.lower().endswith(accept_extension):
                        filepath = os.path.join(root, file)
                        relative_path = os.path.relpath(filepath, directory_path)
                        filename = os.path.splitext(relative_path)[0]
                        matched_files.append(filename)
                        break

        # Sort matched files by directory and filename
        matched_files =  sorted(matched_files)
        return matched_files

    def setup_diffusion_face(self):
        output = {}
        for feature_type in self.feature_types:
            if feature_type == 'light' and not os.path.exists(os.path.join(self.root_dir,f"{feature_type}-anno.txt")):
                scenes = sorted(os.listdir(os.path.join(self.root_dir, "shcoeffs")))
                for scene in scenes:
                    files = sorted(os.listdir(os.path.join(self.root_dir, "shcoeffs", scene)))
                    for filename in files:
                        if filename.endswith(".npy"):
                            shcoeff = np.load(os.path.join(self.root_dir, "shcoeffs", scene, filename))
                            shcoeff = shcoeff.flatten()
                            f_name = filename.replace(".npy","")
                            n_file = f"{scene}/{f_name}"
                            if not n_file in output:
                                output[n_file] = []
                            output[n_file] = shcoeff.tolist()
            else:
                with open(os.path.join(self.root_dir,f"{feature_type}-anno.txt")) as f:
                    lines = f.readlines()
                    for line in lines:
                        contents = line.strip()
                        if len(contents) == 0:
                            continue
                        contents = contents.split(" ")
                        filename = contents[0]
                        if "_" in filename:
                            n_file = filename.split(".")[0]
                        else:
                            file_id = int(filename.split(".")[0])
                            dir_id = int(file_id) // 1000 * 1000
                            n_file = f"{dir_id:05d}/{file_id:05d}"

                        contents = contents[1:]
                        contents = [float(c) for c in contents]
                        if not n_file in output:
                            output[n_file] = []
                        output[n_file] += contents
        # convert to torch tensor 
        for key in output:
            output[key] = torch.from_numpy(np.array(output[key]))
        self.diffusion_face_features = output

    def setup_transform(self):
        self.transform = {}
        # image need to be resize to shape 512x512 and normalize to [-1, 1]
        self.transform['image'] = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])

        self.transform['control'] = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
        ])

        if self.use_background_jitter:
            self.transform['background'] = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.05, hue=0.0),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
                torchvision.transforms.Resize(512,  antialias=True),  # Resize the image to 512x512
            ])
            print("========================")
            print("Using background jitter")
            print("========================")
        else:
            self.transform['background'] = self.transform['image']


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
                if directory == "images":
                    raise Exception("SHOULDNT REACH")
                return image
        raise FileNotFoundError(f"File not found for {name}")
    
    def get_image_files(self):
        files = []
        for fname in sorted(os.listdir(os.path.join(self.root_dir, self.images_dir))):           
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

    def get_background(self, name, height=512, width=512):
        if self.random_mask_background_ratio is None:
            background = self.transform['background'](self.get_image(name, self.backgrounds_dir, height, width))
        elif self.random_mask_background_ratio > 0.0 and self.random_mask_background_ratio <= 1.0:
            background = self.transform['background'](self.get_image(name, self.images_dir, height, width))
            mask = torch.ones((height, width), dtype=torch.float32)
            num_pixels_to_mask = int(height * width * 0.25)
            masked_indices = random.sample(range(height * width), num_pixels_to_mask)
            for idx in masked_indices:
                row, col = divmod(idx, width)
                mask[row, col] = 0
            mask = mask.unsqueeze(0)  # Add channel dimension
            background = background * mask  # Apply mask to image
        elif self.random_mask_background_ratio == 0.0: # just use input image as a background when there is no masking ratio
            background = self.transform['background'](self.get_image(name,self.images_dir, height, width))
        else:
            raise NotImplementedError()

        # convert background from RGB space to AB space if enable    
        if self.use_ab_background: 
            # convert background from [-1,1] to [0,1]
            rgb = (background + 1.0) / 2.0
            assert (rgb >= 0.0).all() and (rgb <=1.0).all() # RGB need to be in format [0,1]

            # convert to numpy format 
            rgb = rgb.permute(1,2,0).numpy()
            
            # convert to rgb image and make sure a and b in range of [-1,1]
            lab = rgb2lab(rgb)
            lab = torch.from_numpy(lab)
            a = lab[...,1] / 128 # convert from [-128,128] to [-1,1]
            b = lab[...,1] / 128

            background = torch.cat([a[None], b[None]],dim=0) # shape [2,H,W]
            assert (rgb >= -1.0).all() and (rgb <=1.0).all()# background is currently in -1,1

        return background 
    
    def get_control_image(self, name:str, directory:str, height=512, width=512):
        try:
            control_image = self.get_image(name, directory, height, width)
        except:
            control_image = torch.zeros(3, 512, 512)
        return control_image

    def shuffle_light_name(self, name):
        """
        Shuffle the name of light direction 
        PS. this only support multi-illumination dataset, will add more support later
        """
        scene_name = name.split("/")[0]
        light_name = name.split("/")[1]
        if not is_valid_light_direction_format_name(light_name):
            raise ValueError(f"Invalid light direction format: {light_name}")

        exist_light_index = extract_combined_number(light_name)
        new_light_index = exist_light_index
        retried_counter = 0
        while new_light_index == exist_light_index and retried_counter < 10:
            new_light_index = random.randint(0, 25)
            retried_counter += 1

        light_name = replace_specific_number_in_text(light_name, exist_light_index, new_light_index)
        new_name = f"{scene_name}/{light_name}"
        return new_name


    def get_item(self, idx, batch_idx):
        name = self.files[idx]
        word_name = self.files[idx]

        image = self.transform['image'](self.get_image(name,self.images_dir, 512, 512))
        background = self.get_background(name, 512, 512)
        shading = self.transform['image'] (self.get_control_image(name,self.shadings_dir, 512, 512))
        if len(self.diffusion_face_features) > 0:
            diffusion_face_features = self.diffusion_face_features[name]
        else:
            diffusion_face_features = []

        if 'light' in self.feature_types and not self.use_shcoeff2:
            # this is for safe-gauard protection in case that accidently pass light feature here
            diffusion_face_features = diffusion_face_features[:-self.light_dimension]


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
            'source_image': image,
            'background': background,
            'shading': shading,
            'diffusion_face': diffusion_face_features,
            'text': prompt,
            'word_name': word_name,
            'idx': idx,
        }
        if self.use_false_shading:
            false_name = self.shuffle_light_name(name)
            output['false_shading'] = self.transform['image'](self.get_control_image(false_name, self.shadings_dir, 512, 512))    
            
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

def is_valid_light_direction_format_name(string):
    """
    Checks if the given string matches the format "dir_{number}_mip2".

    Parameters:
        string (str): The string to check.

    Returns:
        bool: True if the string matches the format, False otherwise.
    """
    pattern = r"^dir_\d+_mip2$"
    return bool(re.match(pattern, string))

def extract_combined_number(text):
    """
    Extracts and combines numbers from the input text.

    Args:
        text (str): Input text containing numbers and other characters.

    Returns:
        int: The combined number from the text.
    """
    # Find all numbers in the text using regex
    numbers = re.findall(r'\d+', text)
    
    # Join the numbers into a single string and convert to integer
    combined_number = int(''.join(numbers))
    return combined_number

def replace_specific_number_in_text(text, target, replacement):
    """
    Replaces a specific number in the input text with the given replacement number.

    Args:
        text (str): Input text containing numbers and other characters.
        target (int): The specific number to replace.
        replacement (int): The number to replace the target number with.

    Returns:
        str: The text with the specific number replaced by the replacement number.
    """
    # Replace the specific target number in the text with the replacement number
    replaced_text = re.sub(rf'\b{target}\b', str(replacement), text)
    return replaced_text
