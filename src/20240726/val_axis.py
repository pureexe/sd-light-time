from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify

#VERSION = 21
VERSIONS = [5]
#CHK_PTS = [4, 9, 14, 19, 24, 44]
CHK_PTS = [4, 9, 14, 19, 24, 29, 44]
LRS = ['1e-4', '5e-4', '5e-5', '3e-4', '8e-5', '1e-4', '5e-4', '5e-5', '1e-4', '5e-4', '5e-5', '3e-4']
NAMES = ['vae_r1', 'vae_r1', 'vae_r1', 'vae_r1', 'vae_r1', 'vae_r2_g0', 'vae_r2_g0', 'vae_r2_g0', 'vae_r1', 'vae_r1', 'vae_r1', 'vae_r1']
#VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']
VAL_FILES = ['light_z_minus', 'light_z_plus', 'light_y_minus', 'light_y_plus', 'light_x_minus', 'light_x_plus']

ENV_MODE = "light_axis"

line = LineNotify()
for version in VERSIONS:
    line.send(f"val_axis.py started at version {version}", with_hostname=True)

    for chk_pt in CHK_PTS:
        CKPT_PATH = f"output/20240726/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={chk_pt:06d}.ckpt"
        model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
        model.set_guidance_scale(3.0)
        #model.set_guidance_scale(0.0)
        model.eval() # disable randomness, dropout, etc...

        if ENV_MODE == "light_direction":
            print("light_direction")
            prompt_path = "datasets/face/face2000_single/prompts.json"
            from EnvmapAffineTestDataset import EnvmapAffineTestDataset
            for axis_name in ['axis_x', 'axis_y', 'axis_z']:
                trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_axis_lightdirection/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/{axis_name}")
                val_dataset = EnvmapAffineTestDataset(root_dir=f"datasets/validation/rotate_point/{axis_name}", dataset_multiplier=10, prompt_path=prompt_path)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

        elif ENV_MODE == "light_axis":
            for val_file in VAL_FILES:
                trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240726/val_axis/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/{val_file}")
                val_dataset = EnvmapAffineDataset(split="0:10", specific_file="split/20k_"+val_file+".json", dataset_multiplier=10, val_hold=0)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
        else:
            raise ValueError(f"Unknown ENV_MODE: {ENV_MODE}")


    line.send(f"val_axis.py at DONE version {version}", with_hostname=True)