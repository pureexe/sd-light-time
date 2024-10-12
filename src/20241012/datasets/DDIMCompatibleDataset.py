from datasets.DDIMDataset import DDIMDataset
import torchvision 
import os
import torch

class DDIMCompatibleDataset(DDIMDataset):
    def get_item(self, idx, batch_idx):
        output = super().get_item(idx, batch_idx)
        output['ldr_envmap'] = output['target_ldr_envmap']
        output['norm_envmap'] = output['target_norm_envmap'] 
        output['word_name'] = output['name'] + '_' +output['word_name']
        return output
    