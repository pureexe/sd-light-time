from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch


#CHK_PTS = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199]
CHK_PTS = [119, 139, 159, 179, 199, 219, 239, 259, 279, 299]
for chk_pt in CHK_PTS:
    CKPT_PATH = f"output/20240703/multi_mlp_fit/lightning_logs/version_8/checkpoints/epoch=000{chk_pt}.ckpt"
    model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
    model.eval() # disable randomness, dropout, etc...

    val_dataset = EnvmapAffineDataset(root_dir="datasets/validation/face10", split="0:10", dataset_multiplier=10, val_hold=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_image/slimnet/5e-5/chk{chk_pt}")
    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)