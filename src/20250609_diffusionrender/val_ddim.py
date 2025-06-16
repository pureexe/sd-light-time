import os
import lightning as L
import torch
import argparse 
import argparse
from constants import FOLDER_NAME

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT

from vll_datasets.DiffusionRendererEnvmapDDIMDataset import DiffusionRendererEnvmapDDIMDataset
from sddiffusionrenderer import SDDiffusionRenderer

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20250221_optmized_shading_exr"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="2")
parser.add_argument("-m", "--mode", type=str, default="val")
parser.add_argument("-g", "--guidance_scale", type=str, default="1")
parser.add_argument("-c", "--checkpoint", type=str, default="lastest")
parser.add_argument("-seed",  type=str, default="42")

args = parser.parse_args()

# we validate 4  model whcih are 
# all
# scrath
# controlnet (SD without adagan)
# adagan


NAMES = {
    7272: 'rank4',
    7273: 'rank16',
    7274: 'rank64',
    117169: 'rank256',
    117170: 'rank4',
    117171: 'rank16',
    117172: 'rank64',
    117173: 'rank256',
}
LRS = {
    7272: '1e-4',
    7273: '1e-4',
    7274: '1e-4',
    117169: '1e-4',
    117170: '1e-5',
    117171: '1e-5',
    117172: '1e-5',
    117173: '1e-5',
}

CHECKPOINTS = {
    7272: 1,
    7273: 1,
    7274: 1,
    117169: 1,
    117170: 1,
    117171: 1,
    117172: 1,
    117173: 1,
}


def get_from_mode(mode):
    if mode == "val":
        return "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val", 100, DiffusionRendererEnvmapDDIMDataset, {"index_file":"/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/index/val.json"},  "a photorealistic image"        

def main():
    print("STARTING VALIDATION...")
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = []
    for checkpoint in args.checkpoint.split(","):
        checkpoint = checkpoint.strip()
        try:
            checkpoint = int(checkpoint)
        except:
            pass 
        checkpoints.append(checkpoint)
    modes = [a.strip() for a in args.mode.split(",")]
    seeds = [int(a.strip()) for a in args.seed.split(",")]

    print("version: ", versions)
    print("guidance_scales: ", guidance_scales)
    print("checkpoints: ", checkpoints)
    print("modes: ", modes)

    for mode in modes:
        for version in versions:
                ddim_class = SDDiffusionRenderer
                try:
                    for checkpoint in checkpoints:
                        dirname = FOLDER_NAME
                        if checkpoint == "lastest":
                            checkpoint = CHECKPOINTS[version]
                        if checkpoint == 0:
                            model = ddim_class(learning_rate=1e-4)
                            CKPT_PATH = None
                        else:
                            CKPT_PATH = f"../../output_t1/{dirname}/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                            if not os.path.exists(CKPT_PATH):
                                print(f"Checkpoint not found: {CKPT_PATH}")
                                continue
                            model = ddim_class.load_from_checkpoint(CKPT_PATH)
                        model.eval() 
                        model.set_num_inference_step(500)
                        output_dir = f"output/{FOLDER_NAME}/val_{mode}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/"
                        # # skip if output dir exist 
                        # if os.path.exists(output_dir):
                        #     print(f"Skip {output_dir}")
                        #     continue
                        os.makedirs(output_dir, exist_ok=True)
                        print("================================")
                        print(output_dir)
                        print("================================")
                        trainer = L.Trainer(max_epochs=1000, precision=MASTER_TYPE, check_val_every_n_epoch=1, default_root_dir=output_dir, inference_mode=False, gradient_clip_val=0)
                        val_root, count_file, dataset_class, dataset_args, specific_prompt = get_from_mode(mode)
     
                        val_dataset = dataset_class( root_dir=val_root,  **dataset_args)
                        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                        trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)


                except Exception as e:
                    raise e

                                
if __name__ == "__main__":
    main()