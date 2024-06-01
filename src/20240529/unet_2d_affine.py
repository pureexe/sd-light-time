import torch
from diffusers.models import UNet2DConditionModel

class UNet2DAffineConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # inject down block 
        # [64,64, 320]
        self.affine_downs = [
            get_MLP(1, 64*64*320*2, 3, 256),
            get_MLP(1, 32*32*640*2, 3, 256),
            get_MLP(1, 16*16*1280*2, 3, 256)
        ]
        self.affine_ups = [
            get_MLP(1, 16*16*1280*2, 3, 256),
            get_MLP(1, 32*32*640*2, 3, 256),
            get_MLP(1, 64*64*320*2, 3, 256)
        ]        # inject up block
        
    def forward(self, x, y):
        x = super().forward(x, y)
        x = self.affine(x)
        return x
    
def get_MLP(n_in, n_out, n_layers, n_hidden):
    layers = []
    for i in range(n_layers):
        layers.append(torch.nn.Linear(n_in, n_hidden))
        layers.append(torch.nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)
