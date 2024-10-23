from datasets.DDIMDataset import DDIMDataset
from datasets.base_relight_dataset import BaseRelightDataset
import torchvision 
import os
import json 

"""
Cross dataset for image_index and envmap_index
"""

class DDIMCrossDataset(BaseRelightDataset):
    def __init__(self, envmap_file, envmap_dir, *args, **kwargs) -> None:
        self.envmap_dir = envmap_dir
        self.envmap_file = envmap_file
        self.envmap_index = []
        self.image_index = []
        if 'image_index' in kwargs:
            self.image_index = kwargs['image_index']
            del kwargs['image_index']
        super().__init__(*args, **kwargs) 
        self.build_index()

    def build_index(self):
        if isinstance(self.envmap_file, dict):
            pass 
        # check if index_file exists
        elif os.path.exists(self.envmap_file):
            with open(self.envmap_file) as f:
                index = json.load(f)
            self.envmap_file = index
        else:
            raise ValueError("index_file should be a dictionary or a path to a file")
        old_files = self.get_image_files() 
        self.files = []
        for envmap in self.envmap_file:
            for filename in old_files:
                self.files.append(filename)
                self.envmap_index.append(envmap)

        self.image_index = self.files
        assert len(self.files) == len(self.envmap_index), "files and envmap_index should have the same length"

    def get_item(self, idx, batch_idx):
        output = super().get_item(idx, batch_idx)
        
        # we will rename both 'ldr_envmap' and 'norm_envmap' to 'source_ldr_envmap' and 'source_norm_envmap'
        output['source_ldr_envmap'] = output.pop('ldr_envmap')
        output['source_norm_envmap'] = output.pop('norm_envmap')

        # we will target_ldr_envmap 
        output['target_ldr_envmap'] = self.transform['envmap'](self.get_image(self.envmap_index[idx],"env_ldr", 256, 256, root_dir = self.envmap_dir))
        output['target_norm_envmap'] = self.transform['envmap'](self.get_image(self.envmap_index[idx],"env_under", 256, 256, root_dir = self.envmap_dir))
        output['word_name'] = self.envmap_index[idx]
        return output

       
    