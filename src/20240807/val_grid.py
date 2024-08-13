# val_grid is a validation at step 

from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
from UnsplashLiteDataset import UnsplashLiteDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import DATASET_ROOT_DIR, FOLDER_NAME


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="0")
parser.add_argument("-m", "--mode", type=str, default="val_z") #unslpash-trainset or multishoe-trainset
parser.add_argument("-g", "--guidance_scale", type=str, default="1,3,5,7")
parser.add_argument("-c", "--checkpoint", type=str, default="9,19,29,39,49") 

#9,19,29,39,49
#199, 399, 599, 799, 999, 1199, 1399, 1599, 1799, 1999, 2199, 2399, 2599, 2799, 2999
args = parser.parse_args()

NAMES = ['unsplash-lite','unsplash-lite','unsplash-lite','unsplash-lite','multishoe','multishoe','multishoe','multishoe']
LRS = ['1e-5', '5e-4', '1e-4', '5e-5', '5e-4', '1e-4', '5e-5', '1e-5']

def get_from_mode(mode):
    if mode == "val_z":
        return "/data/pakkapon/datasets/pointlight_shoe_z_axis", 1000, EnvmapAffineDataset
    elif mode == "sunrise":
        return "/data/pakkapon/datasets/env_shoe_signal_hill_sunrise" , 1000, EnvmapAffineDataset
    elif mode == "unslpash-trainset":
        return "/data/pakkapon/datasets/unsplash-lite/train", 20, UnsplashLiteDataset
    elif mode == "multishoe-trainset":
        return "/data/pakkapon/datasets/pointlight_multishoe/train", 20, UnsplashLiteDataset
    else:
        raise Exception("mode not found")

def main():
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]

    for mode in modes:
        for version in versions:
                for checkpoint in checkpoints:
                     for guidance_scale in guidance_scales:
                        CKPT_PATH = f"output/{FOLDER_NAME}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                        model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
                        model.set_guidance_scale(guidance_scale)
                        model.eval() # disable randomness, dropout, etc...
                        print("================================")
                        print(f"output/{FOLDER_NAME}/val_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}")
                        print("================================")
                        trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/{FOLDER_NAME}/val_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/")
                        val_root, count_file, dataset_calss = get_from_mode(mode)
                        val_dataset = dataset_calss(split=slice(0, count_file, 1), root_dir=val_root)
                        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                        trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

                                
if __name__ == "__main__":
    main()
