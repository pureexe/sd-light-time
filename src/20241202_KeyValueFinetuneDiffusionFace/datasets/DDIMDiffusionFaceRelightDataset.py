# Take array of envmap_index instead

import os 
import json
from datasets.DiffusionFaceRelightDataset import DiffusionFaceRelightDataset

class DDIMDiffusionFaceRelightDataset(DiffusionFaceRelightDataset):

    def __init__(self, index_file, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # check if index_file is dict
        self.index_file = index_file
        self.build_index()
        # now total available files is set to image_index
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

    def get_item(self, idx, batch_idx):

        output = super().get_item(idx, batch_idx)

        # we will rename both 'ldr_envmap' and 'norm_envmap' to 'source_ldr_envmap' and 'source_norm_envmap'
        output['source_diffusion_face'] = output.pop('diffusion_face')
        output['source_background'] = output.pop('background')
        output['source_shading'] = output.pop('shading')

        # here is the thing that will output in format of array 
        output['target_image'] = []
        output['word_name'] = []
 
        output['target_diffusion_face'] = []
        output['target_background'] = []
        output['target_shading'] = []

        for envmap_name in self.envmap_index[idx]:
            output['target_diffusion_face'].append(self.diffusion_face_features[envmap_name]) # this bug on both with or without controlnet, which need to change
            output['target_background'].append(self.transform['image'](self.get_image(envmap_name,"backgrounds", 512, 512)))
            output['target_shading'].append(self.transform['image'](self.get_image(envmap_name,"shadings", 512, 512)))
            output['target_image'].append(self.transform['image'](self.get_image(envmap_name,"images", 512, 512)))
            output['word_name'].append(envmap_name)

        return output
    
        # need a proper way to support new shading :O