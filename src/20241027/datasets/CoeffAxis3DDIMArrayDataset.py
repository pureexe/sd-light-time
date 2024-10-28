import torch
from datasets.Coeff27DDIMArrayDataset import Coeff27DDIMArrayDataset

# we only swap 27 ddim light direction to on the target feature to see if feature still hold

class CoeffAxis3DDIMArrayDataset(Coeff27DDIMArrayDataset):
    def get_mixing_shcoeff(self, source_coeff, target_coeff):
        return torch.cat([source_coeff[:3], target_coeff[3:12],source_coeff[12:]], dim=0)