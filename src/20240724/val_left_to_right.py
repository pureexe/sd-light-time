from RelightDDIMInverse import RelightDDIMInverse
from RelightEnvmapDataset import RelightEnvmapDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify

VERSION = 20
CHK_PTS = [179]
LRS = ['5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '1e-4', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '1e-3', '1e-4', '7e-4', '3e-4', '1e-4', '3e-4', '5e-5']
NAMES = ['dinov2', 'dinov2', 'dinov2', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'vae', 'vae', 'vae', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'slimnet', 'vae5000', 'vae5000', 'vae5000']
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']

ENV_MODE = "mask_left_right"

line = LineNotify()
line.send(f"val_axis.py started at version {VERSION}", with_hostname=True)

for chk_pt in CHK_PTS:
    #CKPT_PATH = f"output/20240703/multi_mlp_fit/lightning_logs/version_{VERSION}/checkpoints/epoch={chk_pt:06d}.ckpt"
    CKPT_PATH = "output/20240726/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000044.ckpt"
    model = RelightDDIMInverse.load_from_checkpoint(CKPT_PATH)
    model.set_guidance_scale(1.0)
    model.eval() # disable randomness, dropout, etc...

    if ENV_MODE == "mask_left_right":
        print("mask_quater")
        prompt_path = "datasets/face/face2000_single/prompts.json"
        from RelightEnvmapDataset import RelightEnvmapDataset
        for axis_name in ['mask_topright']:
            trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240724/relight/left2right/{NAMES[VERSION]}/{LRS[VERSION]}/chk{chk_pt}/")
            val_dataset = RelightEnvmapDataset(index_file="datasets/face/face2000_single/relight_left2right.json")
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

    else:
        raise NotImplementedError("Not implemented yet")

line.send(f"val_axis.py at DONE version {VERSION}", with_hostname=True)
