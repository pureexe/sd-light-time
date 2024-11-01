import torch
from datasets.DDIMArrayEnvDataset import DDIMArrayEnvDataset

# we only swap 27 ddim light direction to on the target feature to see if feature still hold

class Coeff27DDIMArrayDataset(DDIMArrayEnvDataset):

    def get_mixing_shcoeff(self, source_coeff, target_coeff):
        return torch.cat([target_coeff[:27]],[source_coeff[27:]], dim=0)


    def get_item(self, idx, batch_idx):
        output = super().get_item(idx, batch_idx)

        target_sh_coeffs = []
        source_coeff = output['source_sh_coeffs']

        for target_coeff in output['target_sh_coeffs']:
            combined_coeff = self.get_mixing_shcoeff(source_coeff, target_coeff)
            target_sh_coeffs.append(combined_coeff)

        output['target_sh_coeffs'] = target_sh_coeffs

        return output
        