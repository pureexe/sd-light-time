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
CHECKPOINT_FOLDER_NAME = "20241020"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="0")
parser.add_argument("-m", "--mode", type=str, default="multillum_test10_light4")
parser.add_argument("-g", "--guidance_scale", type=str, default="1.0,3.0,5.0,7.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75")
parser.add_argument("-c", "--checkpoint", type=str, default="39")

args = parser.parse_args()
NAMES = {
    0: 'deepfloyd',
}
METHODS = {
    0: 'vae',
}
CONDITIONS_CLASS = {
    0: AffineDepth,
    1: AffineDepthNormalBae,
    2: AffineNormalBae,
    3: AffineNoControl,
    4: AffineNormalBae,
    5: AffineNoControl,
}
LRS = {
    0: '1e-4',
 }
DIRNAME = {
    0: CHECKPOINT_FOLDER_NAME,
}


def get_from_mode(mode):
    if mode == "multillum_test10_light4":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test-10-light-4-array.json"}, "a photo realistic image"
    else:
        raise Exception("mode not found")

def main():
    CONDITIONS_CLASS[0] = AffineNoControl
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]

    for mode in modes:
        for version in versions:
                #condition_class = CONDITIONS_CLASS[version]
                #ddim_class = create_ddim_inversion(condition_class)
                ddim_class = CONDITIONS_CLASS[version]
                # try:
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
                        # disable chromeball inpaint if exist
                        if hasattr(model, 'pipe_chromeball'):
                            del model.pipe_chromeball
                        model.eval() # disable randomness, dropout, etc...
                        model.disable_plot_train_loss()
                        for guidance_scale in guidance_scales:
                            model.set_guidance_scale(guidance_scale)                        
                            output_dir = f"output/{FOLDER_NAME}/val_{mode}/{METHODS[version]}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/"
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


                # except:
                #     pass

                                
if __name__ == "__main__":
    main()