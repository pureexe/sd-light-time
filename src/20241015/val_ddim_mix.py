# val_grid is a validation at step 
import os 
#from RelightDDIMInverse import create_ddim_inversion
from AffineCondition import AffineDepth, AffineNormal, AffineNormalBae, AffineDepthNormal, AffineDepthNormalBae, AffineNoControl

#from DDIMUnsplashLiteDataset import DDIMUnsplashLiteDataset
from datasets.DDIMDataset import DDIMDataset
from datasets.DDIMSingleImageDataset import DDIMSingleImageDataset
from datasets.DDIMCrossDataset import DDIMCrossDataset
from datasets.DDIMSHCoeffsDataset import DDIMSHCoeffsDataset
from datasets.DDIMArrayEnvDataset import DDIMArrayEnvDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20241012"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="2")
parser.add_argument("-m", "--mode", type=str, default="backforth")
parser.add_argument("-g", "--guidance_scale", type=str, default="1,2,3,5,7")
parser.add_argument("-c", "--checkpoint", type=str, default="179")
parser.add_argument("-s", "--split_type", type=str, default="all")

args = parser.parse_args()
NAMES = {
    2: 'depth',
    3: 'bae_both',
    4: 'no_control',
    5: 'bae',
}
METHODS = {
    2: 'vae',
    3: 'vae',
    4: 'vae',
    5: 'vae',
}
CONDITIONS_CLASS = {
    2: AffineDepth,
    3: AffineDepthNormal,
    4: AffineNoControl,
    5: AffineNormal,
}
LRS = {
    2: '1e-4',
    3: '1e-4',
    4: '1e-4',
    5: '1e-4',
 }
DIRNAME = {
    2: CHECKPOINT_FOLDER_NAME,
    3: CHECKPOINT_FOLDER_NAME,
    4: CHECKPOINT_FOLDER_NAME,
    5: CHECKPOINT_FOLDER_NAME,
}


def get_from_mode(mode):
    if mode == "backforth":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test-backforth-relight-array.json"}, "a photo realistic image"
    else:
        raise Exception("mode not found")
    
def get_split_index(split_type, inversion_step):
    if split_type == "all":
        return [True] * inversion_step
    else:
        raise Exception("split type not found")

def main():
    CONDITIONS_CLASS[0] = AffineNoControl
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]

    for mode in modes:
        for version in versions:
                ddim_class = CONDITIONS_CLASS[version]
                #try:
                if True:
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
                        
                        model.eval() # disable randomness, dropout, etc...
                        model.disable_plot_train_loss()
                        for guidance_scale in guidance_scales:
                            model.set_guidance_scale(guidance_scale)                        
                            output_dir = f"output/{FOLDER_NAME}/val_{mode}/{args.split_type}/{METHODS[version]}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/"
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
                            split_index = get_split_index(args.split_type, model.num_inversion_steps)
                            model.set_light_feature_indexs(split_index)
                            if type(count_file) == int:
                                split = slice(0, count_file, 1)
                            else:
                                split = count_file
                            val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)


                # except:
                #     pass

                                
if __name__ == "__main__":
    main()