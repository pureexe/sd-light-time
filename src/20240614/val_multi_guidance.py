from FaceSingeAxisAffine import FaceSingleAxisAffine
from FaceSingleAxisDataset import FaceSingleAxisDataset
import lightning as L
import torch

CKPT_PATH = "output/20240614/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000140.ckpt"
model = FaceSingleAxisAffine.load_from_checkpoint(CKPT_PATH)
model.eval() # disable randomness, dropout, etc...

val_dataset = FaceSingleAxisDataset(split="74:75")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

for guidance_scale in [5,3,1]:
    trainer = L.Trainer(max_epochs =1000, precision=16, check_val_every_n_epoch=1, default_root_dir="output/20240614/multi_fit_val/5e-3/")
    print(f"guidance_scale: {guidance_scale}")
    model.set_guidance_scale(guidance_scale)
    # test (pass in the loader)
    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
