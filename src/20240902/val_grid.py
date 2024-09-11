# val_grid is a validation at step 

from RelightDDIMInverse import create_ddim_inversion
from AffineCondition import AffineDepth, AffineNormal, AffineNormalBae, AffineDepthNormal, AffineDepthNormalBae, AffineNoControl
from UnsplashLiteDataset import UnsplashLiteDataset
from datasets.DDIMCompatibleDataset import DDIMCompatibleDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="29")
parser.add_argument("-m", "--mode", type=str, default="face_left,face_right") #unslpash-trainset or multishoe-trainset
parser.add_argument("-g", "--guidance_scale", type=str, default="3.0")
parser.add_argument("-c", "--checkpoint", type=str, default="19")
args = parser.parse_args()


NAMES = {
    4: 'no_controlnet',
    #4: 'control_depth',
    5: 'control_depth',
    28: 'no_controlnet',
    29: 'no_controlnet',
    30: 'no_controlnet',
    31: 'no_controlnet',
    32: 'no_controlnet',
    33: 'no_controlnet',
}
LRS = {
    4: '1e-4_gate10',
    5: '1e-4_gate100',
    28: '5e-4',
    29: '1e-4',
    30: '5e-5',
    31: '1e-5',
    32: '1e-3',
    33: '5e-3',
}
CONDITIONS_CLASS = {
    4: AffineNoControl,
    5: AffineDepth,
    28: AffineNoControl,
    29: AffineNoControl,
    30: AffineNoControl,
    31: AffineNoControl,
    32: AffineNoControl,
    33: AffineNoControl,
}

def get_from_mode(mode):
    availble_face_prompts = [
        "a color portrait of blond woman with bob hair, white skin, blue eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of man with afro hair-style, tan skin, high quality, photorealistic, center focus",
        "a color portrait of woman with long red hair, white skin, red lips, green eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of young boy with mohawk haircut, white skin, brown eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of chinese woman, dress in traditional Chinese dress,  white skin, brown eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of thai woman, dress in traditional Thai dress, black-long hair, dark skin, brown eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of woman, black-long hair, wearing glasses, white skin, brown eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of girl kid, red hair, wearing fancy headdress, white skin, green eye, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of young boy in green shirt, white skin, wearing middle turkish style Kufi hat, high quality, photorealistic, center focus, looking at the camera",
        "a color portrait of young boy in Thai student uniform with white shirt, high quality, photorealistic, center focus, looking at the camera",
    ]
    if mode == "overfit1":
        return "/data/pakkapon/datasets/unsplash-lite/train_under", slice(0, 1, 1), UnsplashLiteDataset,{}, None
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
    elif mode == "trainset_right" or mode == "trainset_right_flip":
        with open("src/20240815/id_right.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train_under", content, UnsplashLiteDataset,{'is_fliplr': mode == "trainset_right_flip" }, None
    elif mode == "trainset_left" or mode == "trainset_left_flip":
        with open("src/20240815/id_left.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train_under", content, UnsplashLiteDataset,{'is_fliplr': mode == "trainset_left_flip" }, None
    elif mode == "human_left":
        with open("src/20240815/id_left.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train_under", content, UnsplashLiteDataset,{}, availble_face_prompts
    elif mode == "human_right":
        with open("src/20240815/id_right.txt") as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        return "/data/pakkapon/datasets/unsplash-lite/train_under", content, UnsplashLiteDataset,{}, availble_face_prompts
    elif mode == "face_left":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMCompatibleDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-x-minus.json"}, None
    elif mode == "face_right":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMCompatibleDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-x-plus.json"}, None
    else:
        raise Exception("mode not found")
    

def main():
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]
    for mode in modes:
        for version in versions:
                model_class = CONDITIONS_CLASS[version]
                for checkpoint in checkpoints:
                    if checkpoint == 0:
                        model = model_class(learning_rate=1e-4,envmap_embedder='vae', use_consistancy_loss = False)
                        CKPT_PATH = None
                    else:
                        CKPT_PATH = f"output/{FOLDER_NAME}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                        model = model_class.load_from_checkpoint(CKPT_PATH)
                    model.eval() # disable randomness, dropout, etc...
                    if mode in ['face_left']:
                        del model.pipe_chromeball
                    if mode in ['unsplash-trainset-under']:
                        model.enable_plot_train_loss()
                    else:
                        model.disable_plot_train_loss()
                    for guidance_scale in guidance_scales:
                        model.set_guidance_scale(guidance_scale)                        
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