from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="0")
args = parser.parse_args()

#VERSION = 21
VERSIONS = [int(a.strip()) for a in args.version.split(",")]
CHK_PTS = [19,39,59,79,99,119,139,159,179,199,219,239,259,279,299]
LRS = ['split{:02d}'.format(i) for i in range(10)]
NAMES = ['vae_1e-4'] * 10
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']

ENV_MODE = "light_axis"

line = LineNotify()
for version in VERSIONS:
    line.send(f"val_axis.py started at version {version}", with_hostname=True)

    for chk_pt in CHK_PTS:
        CKPT_PATH = f"output/20240730/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={chk_pt:06d}.ckpt"
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
                trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240730/val_axis/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/{val_file}")
                val_dataset = EnvmapAffineDataset(split="0:10", specific_file="split/69k_"+val_file+".json", dataset_multiplier=10, val_hold=0)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
        elif ENV_MODE == "light_axis_self":
            from EnvmapSelfAffineDataset import EnvmapSelfAffineDataset
            for val_file in VAL_FILES:
                trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240730/val_axis_self/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/{val_file}")
                val_dataset = EnvmapSelfAffineDataset(split="0:10", specific_file="split/20k_"+val_file+".json", dataset_multiplier=10, val_hold=0)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
        else:
            raise ValueError(f"Unknown ENV_MODE: {ENV_MODE}")


    line.send(f"val_axis.py at DONE version {version}", with_hostname=True)