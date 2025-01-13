
# bin/siatv100 src/20250104/val_ddim.py -i 95208 -m rotate_copyroom10 -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95209 -m rotate_copyroom10 -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95211 -m rotate_copyroom10 -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95212 -m rotate_copyroom10 -c 40


# bin/siatv100 src/20250104/val_ddim.py -i 95212 -m rotate_copyroom10_balance -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95209 -m rotate_copyroom10_balance -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95208 -m rotate_copyroom10_balance -c 40
# bin/siatv100 src/20250104/val_ddim.py -i 95211 -m rotate_copyroom10_balance -c 40


#bin/siatv100 src/20250104/val_ddim.py -i 95208 -m rotate_copyroom10_left -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95209 -m rotate_copyroom10_left -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95211 -m rotate_copyroom10_left -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95212 -m rotate_copyroom10_left -c 40

#bin/siatv100 src/20250104/val_ddim.py -i 95349 -m rotate_copyroom10 -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95350 -m rotate_copyroom10 -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95352 -m rotate_copyroom10 -c 40
#bin/siatv100 src/20250104/val_ddim.py -i 95354 -m rotate_copyroom10 -c 40


import os
import lightning as L
import torch
import argparse 
#from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sddiffusionface import SDDiffusionFace, ScrathSDDiffusionFace, SDWithoutAdagnDiffusionFace, SDOnlyAdagnDiffusionFace, SDOnlyShading, SDDiffusionFaceNoShading, SDDiffusionFace5ch, SDDiffusionFaceNoBg

from datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20250104"


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
    95208: 'vae_shcoeff',
    95209: 'vae',
    95211: 'clip_shcoeff',
    95212: 'clip',
    95349: 'vae_shcoeff_multi',
    95350: 'vae_multi',
    95352: 'clip_shcoeff_multi',
    95354: 'clip_multi',
}
METHODS = {
    95208: 'default',
    95209: 'default',
    95211: 'default',
    95212: 'default',
    95349: 'default',
    95350: 'default',
    95352: 'default',
    95354: 'default',
}
CONDITIONS_CLASS = {
    95208: SDDiffusionFaceNoBg,
    95209: SDDiffusionFaceNoBg,
    95211: SDDiffusionFaceNoBg,
    95212: SDDiffusionFaceNoBg,
    95349: SDDiffusionFaceNoBg,
    95350: SDDiffusionFaceNoBg,
    95352: SDDiffusionFaceNoBg,
    95354: SDDiffusionFaceNoBg,
}
LRS = {
    95208: '1e-4',
    95209: '1e-4',
    95211: '1e-4',
    95212: '1e-4',
    95349: '1e-4',
    95350: '1e-4',
    95352: '1e-4',
    95354: '1e-4',

}
DIRNAME = {
    95208: CHECKPOINT_FOLDER_NAME,
    95209: CHECKPOINT_FOLDER_NAME,
    95211: CHECKPOINT_FOLDER_NAME,
    95212: CHECKPOINT_FOLDER_NAME,
    95349: CHECKPOINT_FOLDER_NAME,
    95350: CHECKPOINT_FOLDER_NAME,
    95352: CHECKPOINT_FOLDER_NAME,
    95354: CHECKPOINT_FOLDER_NAME,
}
CHECKPOINTS = {
    95208: 40,
    95209: 40,
    95211: 40,
    95212: 40,
    95349: 40,
    95350: 40,
    95352: 40,
    95354: 40,
}

use_ab_background = []
use_shcoeff2 = [95208, 95211, 95349, 95352]
use_only_light = [95208, 95211, 95349, 95352]
use_no_light = [95209, 95212, 95350, 95354]
use_random_mask_background = []

def get_from_mode(mode):
    if mode == "rotate_copyroom10":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_balance":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff_balance", "backgrounds_dir": "control_shading_from_ldr27coeff_balance", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_left_balance":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff_balance", "backgrounds_dir": "control_shading_from_ldr27coeff_balance", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_left":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_everett_kitchen4_left":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_everett_kitchen4_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/everett_kitchen4_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_everett_kitchen4_right":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_everett_kitchen4_right", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/everett_kitchen4_rotate.json", "shadings_dir": "control_shading_from_ldr27coeff", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
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
                            if version in use_shcoeff2:
                                dataset_args['use_shcoeff2'] = True
                            if version in use_only_light:
                                dataset_args['feature_types'] = ['light']
                            if version in use_no_light:
                                dataset_args['feature_types'] = []
                            val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)


                except Exception as e:
                    raise e

                                
if __name__ == "__main__":
    main()