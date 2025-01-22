# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96433 -m interpolate_copyroom10 -c 10
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96433 -m interpolate_copyroom10_static -c 10
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m interpolate_copyroom10_static -c 0
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 1
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 2
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96433 -m all_copyroom10 -c 20
# 
#96458 96461 96462
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 1,2,3,4,5
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 6,7,8,9,10

# all_everett_dining1

# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_copyroom10 -c 1,2,3,4,5
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_copyroom10 -c 6,7,8,9,10
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_copyroom10 -c 11,12,13,14,15
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_copyroom10 -c 16,17,18,19,20 

# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96453 -m all_copyroom10 -c 1,2,3,4,5
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96453 -m all_copyroom10 -c 6,7,8,9,10
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96453 -m all_copyroom10 -c 11,12,13,14,15
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96453 -m all_copyroom10 -c 16,17,18,19,20 


# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 11,12,13
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 14,15
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 16,17,18
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96458 -m all_copyroom10 -c 19,20


# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 11,12,13
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 14,15
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 16,17,18
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 19,20

# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 1,2,3,4,5
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 6,7,8,9,10
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 11,12,13,14,15
# bin/siatv100 src/20250120_efficient_shading/val_ddim.py -i 96434 -m all_everett_dining1 -c 16,17,18,19




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
CHECKPOINT_FOLDER_NAME = "20250120_efficient_shading"


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
    96433: 'vae_shcoeff_1scene',
    96434: 'clip_multiscene',
    96453: 'clip_multiscene',
    96458: 'clip_1scene',
    96461: 'clip_1scene',
    96462: 'clip_1scene',
}
METHODS = {
    96433: 'default',
    96434: 'default',
    96453: 'default',
    96458: 'default',
    96461: 'default',
    96462: 'default',
}
CONDITIONS_CLASS = {
    96433: SDDiffusionFaceNoBg,
    96434: SDDiffusionFaceNoBg,
    96453: SDDiffusionFaceNoBg,
    96458: SDDiffusionFaceNoBg,
    96461: SDDiffusionFaceNoBg,
    96462: SDDiffusionFaceNoBg
}
LRS = {
    96433: '1e-4',
    96434: '1e-4',
    96453: '1e-4',
    96458: '1e-4',
    96461: '1e-5',
    96462: '1e-6'
}
DIRNAME = {
    96433: CHECKPOINT_FOLDER_NAME,
    96434: CHECKPOINT_FOLDER_NAME,
    96453: CHECKPOINT_FOLDER_NAME,
    96458: CHECKPOINT_FOLDER_NAME,
    96461: CHECKPOINT_FOLDER_NAME,
    96462: CHECKPOINT_FOLDER_NAME
}
CHECKPOINTS = {
    96433: 10,
    96434: 10,
    96453: 10,
    96458: 0,
    96461: 10,
    96462: 10
}

use_ab_background = []
use_shcoeff2 = [96433]
use_only_light = [96433]
use_no_light = [96458, 96461, 96462, 96434, 96453]
use_random_mask_background = []

def get_from_mode(mode):
    if mode == "rotate_copyroom10":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_balance":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff_balance", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_left_balance":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff_balance", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_left":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_everett_kitchen4_left":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_everett_kitchen4_left", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/everett_kitchen4_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_everett_kitchen4_right":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_everett_kitchen4_right", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/everett_kitchen4_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "control_shading_from_ldr27coeff", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_right_conv_v2":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
    if mode == "rotate_copyroom10_light0_conv_v2":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_copyroom10_light0", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
    if mode == "interpolate_copyroom10":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_interpolate_copyroom10", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
    if mode == "interpolate_copyroom10_static":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_interpolate_copyroom10_static", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/10n_copyroom10_rotate.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
    if mode == "all_copyroom10":
        return "/data/pakkapon/datasets/multi_illumination/spherical/train", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
    if mode == "all_everett_dining1":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/index/everett_dining1_all.json", "shadings_dir": "control_shading_from_hdr27coeff_conv_v3", "backgrounds_dir": "images", "feature_types": []},  "a photorealistic image"
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