from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch

CKPT_PATH = "output/20240630/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000159.ckpt"
model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
model.eval() # disable randomness, dropout, etc...

val_dataset = EnvmapAffineDataset(split="0:1")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

for guidance_scale in [5]:
    trainer = L.Trainer(max_epochs =1000, precision=16, check_val_every_n_epoch=1, default_root_dir="output/20240630/val_video/5e-5/")
    print(f"guidance_scale: {guidance_scale}")
    model.set_guidance_scale(guidance_scale)
    # test (pass in the loader)
    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
