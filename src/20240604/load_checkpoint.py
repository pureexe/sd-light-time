from LeftRightTimeEmbedding import LeftRightAffine, LeftRightAffineDataset
import lightning as L
import torch

model = LeftRightAffine.load_from_checkpoint("output/20240604_TimeEmbedingV2/lightning_logs/version_0/epoch=266-step=507300_bak.ckpt")

# disable randomness, dropout, etc...
model.eval()


trainer = L.Trainer(max_epochs =1000, precision=32, check_val_every_n_epoch=1, default_root_dir="output/20240604_TimeEmbedingV2_load_ckpt")



val_dataset = LeftRightAffineDataset(split="val")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)



# test (pass in the loader)
trainer.test(model, dataloaders=val_dataloader)
