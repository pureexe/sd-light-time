import torch
from diffusers.models import UNet2DConditionModel

class UNet2DSingleAffineConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_init_dict(config, **kwargs):
        return config, {}, {}
            
    def set_affine(self, affines):
        if len(self.down_blocks) == 6:
            self.down_blocks[0].affine = AffineBlock(affines[0], affines[1])
            self.down_blocks[2].affine = AffineBlock(affines[2], affines[3])
            self.down_blocks[4].affine = AffineBlock(affines[4], affines[5])
        else:
            self.down_blocks.insert(0, AffineBlock(affines[0], affines[1]))
            self.down_blocks.insert(2, AffineBlock(affines[2], affines[3]))
            self.down_blocks.insert(4, AffineBlock(affines[4], affines[5]))

        if len(self.up_blocks) == 6:
            self.up_blocks[0].affine = AffineBlock(affines[6], affines[7])
            self.up_blocks[2].affine = AffineBlock(affines[8], affines[9])
            self.up_blocks[4].affine = AffineBlock(affines[10], affines[11])
        else:
            self.up_blocks.insert(0, AffineBlock(affines[6], affines[7]))
            self.up_blocks.insert(2, AffineBlock(affines[8], affines[9]))
            self.up_blocks.insert(4, AffineBlock(affines[10], affines[11]))
        self.affines = affines
        
class AffineBlock(torch.nn.Module):
    def __init__(self, multipiler, bias):
        super().__init__()
        self.multipiler = multipiler
        self.bias = bias
        
    def forward(self, hidden_states, temb, *args, **kwargs):
        #print(hidden_states.shape)
        #print(self.multipiler.shape)
        out = hidden_states * self.multipiler + self.bias
        resout = out + (hidden_states,)
        return out, resout