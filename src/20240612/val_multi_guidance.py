from FaceLeftRightAffine import FaceLeftRightAffine
from FaceLeftRightDataset import FaceLeftRightDataset
import lightning as L
import torch

model = FaceLeftRightAffine.load_from_checkpoint("output/20240609/multi_fit/lightning_logs/version_1/ep41.ckpt")
model.eval() # disable randomness, dropout, etc...





val_dataset = FaceLeftRightDataset(split="74:75")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

for guidance_scale in [1,3,5]:
    trainer = L.Trainer(max_epochs =1000, precision=32, check_val_every_n_epoch=1, default_root_dir="output/20240609/multi_fit_val/1e-3_ep41_v2/")
    print(f"guidance_scale: {guidance_scale}")
    model.set_guidance_scale(guidance_scale)
    # test (pass in the loader)
    trainer.test(model, dataloaders=val_dataloader)
