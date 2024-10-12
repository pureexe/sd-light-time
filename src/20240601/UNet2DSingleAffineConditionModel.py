import torch
from diffusers.models import UNet2DConditionModel

class UNet2DSingleAffineConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_init_dict(config, **kwargs):
        return config, {}, {}
            
    def set_affine(self, affines):
        for block_id in range(3):
            if len(self.down_blocks[block_id].downsamplers) == 1:
                self.down_blocks[block_id].downsamplers.insert(0, AffineBlock(affines[block_id*2], affines[block_id*2+1], id=block_id, type='down'))
            else:
                self.down_blocks[block_id].downsamplers[0] = AffineBlock(affines[block_id*2], affines[block_id*2+1])

        for block_id in range(3):
            if len(self.up_blocks[block_id].upsamplers) == 1:
                self.up_blocks[block_id].upsamplers.insert(0, AffineBlock(affines[block_id*2+6], affines[block_id*2+7], id=block_id, type='down'))
            else:
                self.up_blocks[block_id].upsamplers[0] = AffineBlock(affines[block_id*2+6], affines[block_id*2+7])

    def get_affine_params(self):
        # return pytorch parameter for affine block
        all_params = []
        for block_id in range(3):
            all_params.extend(
                self.down_blocks[block_id].downsamplers[0].parameters()
            )
        for block_id in range(3):
            all_params.extend(self.up_blocks[block_id].upsamplers[0].parameters())
        return all_params
    
        
class AffineBlock(torch.nn.Module):
    def __init__(self, multipiler, bias, id=0, type='down'):
        super().__init__()
        self.multipiler = torch.nn.Parameter(multipiler)
        self.bias = torch.nn.Parameter(bias)
        self.id = id
        self.type = type
        
    def forward(self, hidden_states, *args, **kwargs):
        out = hidden_states * self.multipiler + self.bias
        return out