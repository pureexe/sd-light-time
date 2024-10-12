from FaceLeftRightAffine import FaceLeftRightAffine
from FaceLeftRightDataset import FaceLeftRightDataset
import lightning as L
import torch

model = FaceLeftRightAffine.load_from_checkpoint("output/20240609/single_fit/lightning_logs/version_2/ep20.ckpt")
model.eval() # disable randomness, dropout, etc...


trainer = L.Trainer(max_epochs =1000, precision=32, check_val_every_n_epoch=1, default_root_dir="output/20240609/single_fit_val")


val_dataset = FaceLeftRightDataset(split="74:75")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# test (pass in the loader)
trainer.test(model, dataloaders=val_dataloader)
