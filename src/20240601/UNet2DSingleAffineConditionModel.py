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
                self.down_blocks[block_id].downsamplers.insert(0, AffineBlock(affines[block_id*2], affines[block_id*2+1]))
            else:
                self.down_blocks[block_id].downsamplers[0] = AffineBlock(affines[block_id*2], affines[block_id*2+1])

        for block_id in range(3):
            if len(self.up_blocks[block_id].upsamplers) == 1:
                self.up_blocks[block_id].upsamplers.insert(0, AffineBlock(affines[block_id*2+6], affines[block_id*2+7]))
            else:
                self.up_blocks[block_id].upsamplers[0] = AffineBlock(affines[block_id*2+6], affines[block_id*2+7])
        
class AffineBlock(torch.nn.Module):
    def __init__(self, multipiler, bias):
        super().__init__()
        self.multipiler = multipiler
        self.bias = bias
        
    def forward(self, hidden_states, *args, **kwargs):
        #print(self.multipiler)
        out = hidden_states #* self.multipiler + self.bias
        return out