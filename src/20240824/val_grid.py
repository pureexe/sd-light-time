# val_grid is a validation at step 

from AffineConsistancy import AffineConsistancy
from UnsplashLiteDataset import UnsplashLiteDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="1")
#parser.add_argument("-m", "--mode", type=str, default="human_left") #unslpash-trainset or multishoe-trainset
parser.add_argument("-m", "--mode", type=str, default="unsplash-trainset-under")
parser.add_argument("-g", "--guidance_scale", type=str, default="1.0")
parser.add_argument("-c", "--checkpoint", type=str, default=",".join([str(i) for i in range(32,61)]) ) 
#parser.add_argument("-g", "--guidance_scale", type=str, default="1.0,3.0")
#parser.add_argument("-c", "--checkpoint", type=str, default=",".join([str(i) for i in range(30, 33)]) ) 
#parser.add_argument("-c", "--checkpoint", type=str, default="35,30,25,20,15,10,5" )
#parser.add_argument("-c", "--checkpoint", type=str, default="20" )

#9,19,29,39,49
#199, 399, 599, 799, 999, 1199, 1399, 1599, 1799, 1999, 2199, 2399, 2599, 2799, 2999
args = parser.parse_args()
NAMES = ['new_light_block', 'new_light_block', 'new_light_block', 'new_light_block']
LRS = ['5e-4', '1e-4', '5e-5', '1e-5' ]

def get_from_mode(mode):
    availble_face_prompts = [
        "a photo of woman face",
        "a photo of man face",
        "a photo of boy face",
        "a photo of girl face",
        "a photo of joyful face",
        "a photo of cheerful face",
        "a photo of thoughtful face",
        "a photo of crying face",
        "a photo of angry face",
        "a photo of disgusted face",
    ]
    if mode == "z":
        return "/data/pakkapon/datasets/pointlight_shoe_z/validation", 60, UnsplashLiteDataset,{}, None
    elif mode == "sunrise":
        return "/data/pakkapon/datasets/sunrise/validation", 60, UnsplashLiteDataset,{}, None
    elif mode == "unsplash-trainset-under":
        return "/data/pakkapon/datasets/unsplash-lite/train_under", 10, UnsplashLiteDataset,{}, None
    elif mode == "unsplash-trainset-norm":
        return "/data/pakkapon/datasets/unsplash-lite/train_under", 10, UnsplashLiteDataset,{}, None
    elif mode == "multishoe-trainset":
        return "/data/pakkapon/datasets/pointlight_multishoe/train", 20, UnsplashLiteDataset,{}, None
    elif mode == "rainbow":
            return "/data/pakkapon/datasets/rainbow-backgroud/validation", 100, UnsplashLiteDataset,{}, None
    elif mode == "morning_cat":
        return "/data/pakkapon/datasets/morning_cat/validation", 60, UnsplashLiteDataset,{}, None
    elif mode == "human_left":
        with open("src/20240815/id_left.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train", content, UnsplashLiteDataset,{}, availble_face_prompts
    elif mode == "human_right":
        with open("src/20240815/id_right.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train", content, UnsplashLiteDataset,{}, availble_face_prompts
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
                        if checkpoint == 0:
                            model = AffineConsistancy(learning_rate=1e-4,envmap_embedder='vae', use_consistancy_loss = False)
                            CKPT_PATH = None
                        else:
                            CKPT_PATH = f"output/20240824/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                            model = AffineConsistancy.load_from_checkpoint(CKPT_PATH)
                        model.set_guidance_scale(guidance_scale)
                        model.eval() # disable randomness, dropout, etc...
                        print("================================")
                        print(f"output/{FOLDER_NAME}/val_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}")
                        print("================================")
                        trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/{FOLDER_NAME}/val_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/")
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