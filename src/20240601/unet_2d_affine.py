from diffusers.models import UNet2DConditionModel

class UNet2DAffineConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # inject down block 
        # [64,64, 320]
        # [32, 32, 640]
        # [16, 16, 1280]

        # up 
        # [16, 16, 1280]
        # [32, 32, 640]
        # [64,64,320]

        # inject up block
        
    def forward(self, x, y):
        x = super().forward(x, y)
        x = self.affine(x)
        return x
