import os
import lightning as L
import torch
import argparse 
import argparse
from constants import FOLDER_NAME
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT

from vll_datasets.DiffusionRendererEnvmapDDIMDataset import DiffusionRendererEnvmapDDIMDataset
from sdrelightenv import SDRelightEnv, SDAlbedoNormalDepthRelightEnv

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20250614_light_embed_condition"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="7406")
parser.add_argument("-m", "--mode", type=str, default="val")
parser.add_argument("-g", "--guidance_scale", type=str, default="1")
parser.add_argument("-c", "--checkpoint", type=str, default="7")
parser.add_argument("-seed",  type=str, default="42")

args = parser.parse_args()

STEP_PER_CHECKPOINT = 770

NAMES = {
    7450: 'anti_shock_rank16',
}
LRS = {
    7450: '1e-4',
}
CHECKPOINTS = {
    7450: 1,
}
DDIM_CLASSES = {
    7450: SDRelightEnv
}


def get_from_mode(mode):
    if mode == "val":
        return "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val", 100, DiffusionRendererEnvmapDDIMDataset, {"index_file":"/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/index/val.json"},  "a photorealistic image"
    if mode == "val_identity":
        return "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val", 100, DiffusionRendererEnvmapDDIMDataset, {"index_file":"/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/index/val_identity.json"},  "a photorealistic image"        

def get_ddim_class(version):
    if version not in DDIM_CLASSES:
        raise ValueError(f"Version {version} not supported.")
    return DDIM_CLASSES[version]

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

    dirname = FOLDER_NAME
    gates = []
    gates.append(np.zeros((17,)))
    for version in versions:
        ddim_class = get_ddim_class(version)
        for checkpoint in checkpoints:
            CKPT_PATH = f"../../output_t1/{dirname}/lightning_logs/version_{version}/checkpoints/epoch={checkpoint-1}-step={checkpoint*770}.ckpt"
            model = ddim_class.load_from_checkpoint(CKPT_PATH)
            model.eval() 
            gates.append(model.attention_light_gate[:17].cpu().detach().numpy())

    # plot gates using matplotlib
    gates = np.array(gates)
    print(gates.shape)
    
    plt.figure(figsize=(10, 6))
    for i in range(1, gates.shape[1]):
        plt.plot(np.arange(gates.shape[0]), gates[:,i], label=f'Gate {i}', marker='o', linestyle='-')
    plt.title('Value of Light Attention Gates')
    plt.xlabel('Epoch')
    plt.ylabel('Gate Value')
    plt.legend()
    output_dir = f"../../output_t1/{dirname}/lightning_logs/version_{version}/gates"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'gates.png'))
    print("SAVE VALUE")
    exit()



    for mode in modes:
        for version in versions:
                ddim_class = get_ddim_class(version)
                try:
                    for checkpoint in checkpoints:
                        dirname = FOLDER_NAME
                        if checkpoint == "lastest":
                            checkpoint = CHECKPOINTS[version]
                        if checkpoint == 0:
                            model = ddim_class(learning_rate=1e-4)
                            CKPT_PATH = None
                        else:
                            
                            if not os.path.exists(CKPT_PATH):
                                print(f"Checkpoint not found: {CKPT_PATH}")
                                continue
                            model = ddim_class.load_from_checkpoint(CKPT_PATH)
                        model.eval() 
                        model.set_num_inference_step(500)
                        output_dir = f"../../output_t1/{FOLDER_NAME}/val_{mode}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/"
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