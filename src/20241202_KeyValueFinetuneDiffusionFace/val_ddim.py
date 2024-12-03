
# validation 
# bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/val_ddim.py -i 91829 -m valid2left
# bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/val_ddim.py -i 91829 -m valid2right
# bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/val_ddim.py -i 91829 -m train2left
# bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/val_ddim.py -i 91829 -m train2right
# bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/val_ddim.py -i 91829 -m viz_v2


import os 

import lightning as L
import torch
import argparse 
#from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sdKeyValueFinetune import SDKeyValueFinetune, SDKeyValueFinetuneWithoutControlNet

from datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20241202_KeyValueFinetuneDiffusionFace"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="2")
parser.add_argument("-m", "--mode", type=str, default="face_left,face_right")
parser.add_argument("-g", "--guidance_scale", type=str, default="1")
parser.add_argument("-c", "--checkpoint", type=str, default="lastest")

args = parser.parse_args()

# we validate 4  model whcih are 
# all
# scrath
# controlnet (SD without adagan)
# adagan


NAMES = {
    91829: 'mint_pretrain', #all 1e-4
    91830: 'mint_pretrain', #all 1e-5
}
METHODS = {
    91829: 'default',
    91830: 'default'
}
CONDITIONS_CLASS = {
    91829: SDKeyValueFinetune,
    91830: SDKeyValueFinetune
}
LRS = {
    91829: '1e-4',
    91830: '1e-5'
}
DIRNAME = {
    91829: CHECKPOINT_FOLDER_NAME,
    91830: CHECKPOINT_FOLDER_NAME,
}
CHECKPOINTS = {
    91829: 5,
    91830: 4,
}


def get_from_mode(mode):
    if mode == "face_left":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid/split-x-minus-array.json"}, "a photorealistic image"
    elif mode == "face_right":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid/split-x-plus-array.json"}, "a photorealistic image"
    elif mode == "viz_v2":
        return "/data/pakkapon/datasets/face/ffhq_defareli/viz", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/viz/val-viz-array.json"}, "a photorealistic image"
    elif mode == "train_left":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train/split-x-minus-array.json"}, "a photorealistic image"
    elif mode == "train_right":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train/split-x-plus-array.json"}, "a photorealistic image"
    elif mode == "train_left_v2":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train/split-x-minus-array.json"}, "a photorealistic image"
    elif mode == "train_right_v2":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train/split-x-plus-array.json"}, "a photorealistic image"
    elif mode == "train2left":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train2left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train2left/index-array.json"}, "a photorealistic image"
    elif mode == "train2right":
        return "/data/pakkapon/datasets/face/ffhq_defareli/train2right", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/train2right/index-array.json"}, "a photorealistic image"
    elif mode == "valid2left":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid2left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid2left/index-array.json"}, "a photorealistic image"
    elif mode == "valid2right":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid2right", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid2right/index-array.json"}, "a photorealistic image"
    else:
        raise Exception("mode not found")

def main():
    print("STARTING VALIDATION...")
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    #checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    checkpoints = []
    for checkpoint in args.checkpoint.split(","):
        checkpoint = checkpoint.strip()
        try:
            checkpoint = int(checkpoint)
        except:
            pass 
        checkpoints.append(checkpoint)
    modes = [a.strip() for a in args.mode.split(",")]

    print("version: ", versions)
    print("guidance_scales: ", guidance_scales)
    print("checkpoints: ", checkpoints)
    print("modes: ", modes)

    for mode in modes:
        for version in versions:
                ddim_class = CONDITIONS_CLASS[version]
                try:
                    for checkpoint in checkpoints:
                        dirname = DIRNAME[version]
                        if checkpoint == "lastest":
                            checkpoint = CHECKPOINTS[version]
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


                except Exception as e:
                    raise e

                                
if __name__ == "__main__":
    main()