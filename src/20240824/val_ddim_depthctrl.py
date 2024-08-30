# val_grid is a validation at step 

from RelightDDIMDepthInverse import RelightDDIMDepthInverse

from DDIMUnsplashLiteDataset import DDIMUnsplashLiteDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="1")
parser.add_argument("-m", "--mode", type=str, default="left2right") #unslpash-trainset or multishoe-trainset
parser.add_argument("-g", "--guidance_scale", type=str, default="3.0")
parser.add_argument("-c", "--checkpoint", type=str, default="80")

args = parser.parse_args()
NAMES = ['new_light_block', 'new_light_block', 'new_light_block', 'new_light_block', 'gate10', 'gate100']
LRS = ['5e-4', '1e-4', '5e-5', '1e-5', '1e-4', '1e-4']
    

def get_from_mode(mode):
    if mode == "left2right":
        return "/data/pakkapon/datasets/unsplash-lite/train", 1, DDIMUnsplashLiteDataset,{'index_file': 'src/20240824/ddim_left2right100.json'}, None
    elif mode == "left2right_dev":
        return "/data/pakkapon/datasets/unsplash-lite/train", 1, DDIMUnsplashLiteDataset,{'index_file': 'src/20240824/ddim_dev.json'}, None
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
                    if checkpoint == 0:
                        model = RelightDDIMDepthInverse(learning_rate=1e-4,envmap_embedder='vae', use_consistancy_loss = False)
                        CKPT_PATH = None
                    elif version == 4:
                        CKPT_PATH = f"output/20240826/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch={checkpoint:06d}.ckpt"
                        raise Exception("Not implemented")
                        from RelightDDIMInverse26 import RelightDDIMInverse26
                        model = RelightDDIMInverse26.load_from_checkpoint(CKPT_PATH)
                    elif version == 5:
                        CKPT_PATH = f"output/20240826/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch={checkpoint:06d}.ckpt"
                        raise Exception("Not implemented")
                        from RelightDDIMInverse26 import RelightDDIMInverse26
                        model = RelightDDIMInverse26.load_from_checkpoint(CKPT_PATH)
                    else:
                        CKPT_PATH = f"output/20240824/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                        model = RelightDDIMDepthInverse.load_from_checkpoint(CKPT_PATH)
                    model.eval() # disable randomness, dropout, etc...
                    model.disable_plot_train_loss()
                    for guidance_scale in guidance_scales:
                        model.set_guidance_scale(guidance_scale)                        
                        output_dir = f"output/{FOLDER_NAME}/val_ddim_depthctrl_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}"
                        print("================================")
                        print(output_dir)
                        print("================================")
                        trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=output_dir)
                        val_root, count_file, dataset_class, dataset_args, specific_prompt = get_from_mode(mode)
                        if type(count_file) == int:
                            split = slice(0, count_file, 1)
                        else:
                            split = count_file
                        val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                        trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

                                
if __name__ == "__main__":
    main()