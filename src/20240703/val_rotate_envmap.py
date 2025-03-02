from EnvmapAffineValRotate import EnvmapAffineValRotate
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify

VERSION = 20
CHK_PTS = [179]
LRS = ['5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '1e-4', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '1e-3', '1e-4', '7e-4', '3e-4', '1e-4', '3e-4', '5e-5']
NAMES = ['dinov2', 'dinov2', 'dinov2', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'vae', 'vae', 'vae', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'slimnet', 'vae5000', 'vae5000', 'vae5000']
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']
ENV_MODE = "rotate_envmap"

line = LineNotify()
line.send(f"val_rotate_envmap.py started at version {VERSION}", with_hostname=True)

for chk_pt in CHK_PTS:
    CKPT_PATH = f"output/20240703/multi_mlp_fit/lightning_logs/version_{VERSION}/checkpoints/epoch={chk_pt:06d}.ckpt"
    model = EnvmapAffineValRotate.load_from_checkpoint(CKPT_PATH)
    model.set_guidance_scale(3.0)
    model.eval() # disable randomness, dropout, etc...

    if ENV_MODE == "rotate_envmap":
        print("rotate_envmap")
        #for val_file in VAL_FILES:
        for val_file in [VAL_FILES[5]]:
            trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_rotate_envmap/{NAMES[VERSION]}/{LRS[VERSION]}/chk{chk_pt}/{val_file}")
            val_dataset = EnvmapAffineDataset(split="0:10", specific_file=""+val_file+".json", dataset_multiplier=10, val_hold=0)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

    else:
        raise ValueError("Unknown ENV_MODE")
    
line.send(f"val_rotate_envmap.py at DONE version {VERSION}", with_hostname=True)