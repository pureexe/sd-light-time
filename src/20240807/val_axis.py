from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import DATASET_ROOT_DIR, FOLDER_NAME

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="0,1")
parser.add_argument("-g", "--guidance_scale", type=str, default="1,3,5,7")
args = parser.parse_args()

#VERSION = 21
VERSIONS = [int(a.strip()) for a in args.version.split(",")]
#CHK_PTS = [19,39,59,79,99,119,139,159,179,199,219,239,259,279,299]
#CHK_PTS = [999,999,999, 999, 99, 99, 99, 99]\
#CHK_PTS = [99]
CHK_PTS = [9]

LRS = ['1e-5', '5e-4']
NAMES = ['vae'] * 20
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']
GUIDANCE_SCALE = [float(a.strip()) for a in args.guidance_scale.split(",")]
VAL_MODE = "z"
#ENV_MODE = "sunrise"
ENV_MODE = "light_val_all"
VAL_Y_ROOT_DIR = "/data/pakkapon/datasets/pointlight_shoe_y_axis"
VAL_Z_ROOT_DIR = "/data/pakkapon/datasets/pointlight_shoe_z_axis"
AROUND_ROOT_DIR = "/data/pakkapon/datasets/pointlight_around"
SUNRISE_ROOT_DIR = "/data/pakkapon/datasets/env_shoe_signal_hill_sunrise"
NAME = "unsplash20k"


line = LineNotify()
for version in VERSIONS:
    # if version in [1]:
    #     NAME = "shoe4light250"
    # if version in [8,9,10,11]:
    #     NAME = "rand100"
        
    # if version in [4,5,6,7]:
    #         NAME = "rand1000"
    
    line.send(f"val_axis.py started at version {version}", with_hostname=True)

    for chk_pt in CHK_PTS:
        for guidance_scale in GUIDANCE_SCALE:
            CKPT_PATH = f"output/{FOLDER_NAME}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={chk_pt:06d}.ckpt"
            model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
            #model.set_guidance_scale(3.0)
            model.set_guidance_scale(guidance_scale)
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
            elif ENV_MODE == "around":
                if True:
                    trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240801/val_{NAME}_around/{GUIDANCE_SCALE}/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/")
                    val_root = AROUND_ROOT_DIR
                    val_dataset = EnvmapAffineDataset(split=slice(0, 360, 1), root_dir=val_root)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
            elif ENV_MODE == "sunrise":
                if True:
                    trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/{FOLDER_NAME}/val_{NAME}_sunrise/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/")
                    val_root = SUNRISE_ROOT_DIR
                    val_dataset = EnvmapAffineDataset(split=slice(0, 360, 1), root_dir=val_root)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
            elif ENV_MODE == "light_val_all":
                if True:
                    print("================================")
                    print(f"output/{FOLDER_NAME}/val_{VAL_MODE}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{chk_pt}")
                    print("================================")
                    trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1,    default_root_dir=f"output/{FOLDER_NAME}/val_{VAL_MODE}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{chk_pt}/")
                    if VAL_MODE == "z":
                        val_root = VAL_Z_ROOT_DIR
                    else:
                        val_root = VAL_Y_ROOT_DIR
                    val_dataset = EnvmapAffineDataset(split=slice(0, 360, 1), root_dir=val_root)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                    trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
            else:
                raise ValueError(f"Unknown ENV_MODE: {ENV_MODE}")


    line.send(f"val_axis.py at DONE version {version}", with_hostname=True)