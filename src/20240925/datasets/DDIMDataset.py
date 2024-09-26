
import os 
import json
from datasets.base_relight_dataset import BaseRelightDataset

class DDIMDataset(BaseRelightDataset):
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
        output['source_ldr_envmap'] = output.pop('ldr_envmap')
        output['source_norm_envmap'] = output.pop('norm_envmap')
        output['source_sh_coeffs'] = output.pop('sh_coeffs')
        

        # we will target_ldr_envmap 
        output['target_ldr_envmap'] = self.transform['envmap'](self.get_image(self.envmap_index[idx],"env_ldr", 256, 256))
        output['target_norm_envmap'] = self.transform['envmap'](self.get_image(self.envmap_index[idx],"env_under", 256, 256))
        output['target_image'] = self.transform['image'](self.get_image(self.envmap_index[idx], "images", 512, 512))
        output['target_sh_coeffs'] = self.get_shcoeffs(self.envmap_index[idx]).flatten()
        output['word_name'] = self.envmap_index[idx]
        return output

    def check_output(self, output):
        for key in ['name', 'word_name', 'source_ldr_envmap', 'source_norm_envmap', 'target_ldr_envmap', 'target_norm_envmap']:
            assert key in output, f"{key} is not in output"
        return True
