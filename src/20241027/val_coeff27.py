# val_grid is a validation at step 
import os 
#from RelightDDIMInverse import create_ddim_inversion
from AffineCondition import AffineDepth, AffineNormal, AffineNormalBae, AffineDepthNormal, AffineDepthNormalBae, AffineNoControl

#from DDIMUnsplashLiteDataset import DDIMUnsplashLiteDataset
from datasets.Coeff27DDIMArrayDataset import Coeff27DDIMArrayDataset
from datasets.DDIMArrayEnvDataset import DDIMArrayEnvDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20241027"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="87186")
parser.add_argument("-m", "--mode", type=str, default="face60k_fuse_train3")
parser.add_argument("-g", "--guidance_scale", type=str, default="1.0")
parser.add_argument("-c", "--checkpoint", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,29,20")

args = parser.parse_args()
NAMES = {
    87186: 'no_control',
    87187: 'no_control',
    87188: 'no_control',
    87189: 'no_control',
}
METHODS = {
    87186: 'no_control',
    87187: 'no_control',
    87188: 'no_control',
    87189: 'no_control',
}
CONDITIONS_CLASS = {
    87186: AffineNoControl,
    87187: AffineNoControl,
    87188: AffineNoControl,
    87189: AffineNoControl,
}
LRS = {
    87186: 1e-4,
    87187: 5e-5,
    87188: 1e-5,
    87189: 5e-4,
 }
DIRNAME = {
    87186: CHECKPOINT_FOLDER_NAME,
    87187: CHECKPOINT_FOLDER_NAME,
    87188: CHECKPOINT_FOLDER_NAME,
    87189: CHECKPOINT_FOLDER_NAME,
}


def get_from_mode(mode):
    if mode == "face60k_fuse_train3":
        return "/data/pakkapon/datasets/face/face60k_fuse", 100, Coeff27DDIMArrayDataset,{"index_file":"/data/pakkapon/datasets/face/face60k/train-viz-array.json"}, "a photorealistic image"
    elif mode == "faceval10k_fuse_test":
        return "/data/pakkapon/datasets/face/faceval10k_fuse", 100, Coeff27DDIMArrayDataset,{"index_file":"/data/pakkapon/datasets/face/faceval10k_fuse/val-viz-array.json"}, "a photorealistic image"
    elif mode == "faceval10k_fuse_test_left":
        return "/data/pakkapon/datasets/face/faceval10k_fuse", 100, Coeff27DDIMArrayDataset,{"index_file":"/data/pakkapon/datasets/face/faceval10k_fuse/split-x-minus-array.json"}, "a photorealistic image"
    elif mode == "faceval10k_fuse_test_right":
        return "/data/pakkapon/datasets/face/faceval10k_fuse", 100, Coeff27DDIMArrayDataset,{"index_file":"/data/pakkapon/datasets/face/faceval10k_fuse/split-x-plus-array.json"}, "a photorealistic image"
    else:
        raise Exception("mode not found")

def main():
    print("STARTING VALIDATION...")
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]

    for mode in modes:
        for version in versions:
                ddim_class = CONDITIONS_CLASS[version]
                try:
                    for checkpoint in checkpoints:
                        dirname = DIRNAME[version]
                        if checkpoint == 0:
                            model = ddim_class(learning_rate=1e-4)
                            CKPT_PATH = None
                        else:
                            CKPT_PATH = f"output/{dirname}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                            if not os.path.exists(CKPT_PATH):
                                print(f"Checkpoint not found: {CKPT_PATH}")
                                continue
                            model = ddim_class.load_from_checkpoint(CKPT_PATH)
                        # disable chromeball inpaint if exist
                        if hasattr(model, 'pipe_chromeball'):
                            del model.pipe_chromeball
                        model.eval() # disable randomness, dropout, etc...
                        model.disable_plot_train_loss()
                        for guidance_scale in guidance_scales:
                            model.set_guidance_scale(guidance_scale)                        
                            output_dir = f"output/{FOLDER_NAME}/val_coeff27_{mode}/{METHODS[version]}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/"
                            # skip if output dir exist 
                            if os.path.exists(output_dir):
                                print(f"Skip {output_dir}")
                                continue
                            os.makedirs(output_dir, exist_ok=True)
                            print("================================")
                            print(output_dir)
                            print("================================")
                            trainer = L.Trainer(max_epochs=1000, precision=MASTER_TYPE, check_val_every_n_epoch=1, default_root_dir=output_dir, inference_mode=False, gradient_clip_val=0)
                            val_root, count_file, dataset_class, dataset_args, specific_prompt = get_from_mode(mode)
                            if type(count_file) == int:
                                split = slice(0, count_file, 1)
                            else:
                                split = count_file
                            val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)


                except Exception as e:
                    raise e

                                
if __name__ == "__main__":
    main()