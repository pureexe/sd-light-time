import os 
import torch 
import torchvision 
import json
import ezexr 
import numpy as np

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
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.dataset_multiplier = dataset_multiplier
        self.prompt = self.get_prompt_from_file(prompt_file) 
        self.specific_prompt = specific_prompt
        self.setup_transform()
        self.setup_diffusion_face()
        # setup image index
        if index_file != "" and index_file != None:
            self.index_file = kwargs['index_file']
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
            directory_path = os.path.join(self.root_dir, "images")

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
        feature_types = ['shape', 'cam', 'faceemb', 'shadow']
        output = {}
        for feature_type in feature_types:
            with open(os.path.join(self.root_dir,f"{feature_type}-anno.txt")) as f:
                lines = f.readlines()
                for line in lines:
                    contents = line.strip()
                    if len(contents) == 0:
                        continue
                    contents = contents.split(" ")
                    filename = contents[0]
                    file_id = int(filename.split(".")[0])
                    contents = contents[1:]
                    contents = [float(c) for c in contents]
                    dir_id = int(file_id) // 1000 * 1000
                    n_file = f"{dir_id:05d}/{file_id:05d}"
                    if not n_file in output:
                        output[n_file] = []
                    output[n_file] += contents
        # convert to torch tensor 
        for key in output:
            output[key] = torch.from_numpy(np.array(output[key])).float()
        self.diffusion_face_features = output

    def setup_transform(self):
        self.transform = {}
        # image need to be resize to shape 512x512 and normalize to [-1, 1]
        self.transform['image'] = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
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


    def get_item(self, idx, batch_idx):
        name = self.files[idx]
        word_name = self.files[idx]

        image = self.transform['image'](self.get_image(name,"images", 512, 512))
        background = self.transform['image'](self.get_image(name,"backgrounds", 512, 512))
        shading = self.transform['image'](self.get_image(name,"shadings", 512, 512))
        diffusion_face_features = self.diffusion_face_features[name]

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