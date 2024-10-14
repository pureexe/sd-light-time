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
import numpy as np

CHECKPOINT_FOLDER_NAME = "20240918"


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="33")
parser.add_argument("-m", "--mode", type=str, default="multillum_test30_strength")
parser.add_argument("-g", "--guidance_scale", type=str, default="1,1.5,2,2.5,3,3.5,5,7")
parser.add_argument("-c", "--checkpoint", type=str, default="499")
#parser.add_argument("--strength", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
parser.add_argument("-s", "--strength", type=str, default="0.2,0.4,0.6,0.8,1.0")

args = parser.parse_args()
NAMES = {
    0: 'no_control',
    1: 'depth',
    2: 'normal',
    3: 'both',
    4: 'no_control',
    5: 'depth',
    6: 'normal',
    7: 'both',
    8: 'bae',
    9: 'bae_both',
    10: 'bae',
    11: 'bae_both',
    12: 'no_control',
    13: 'depth',
    14: 'normal',
    15: 'both',
    16: 'no_control',
    17: 'depth',
    18: 'normal',
    19: 'both',
    20: 'bae',
    21: 'bae_both',
    22: 'bae',
    23: 'bae_both',
    24: 'no_control',
    25: 'depth',
    26: 'bae_both',
    27: 'bae',
    33: 'no_control',
    35: 'both_bae',
    36: 'bae',
    37: 'depth'
}
METHODS = {
    12: 'shcoeffs',
    13: 'shcoeffs',
    14: 'shcoeffs',
    15: 'shcoeffs',
    16: 'shcoeffs',
    17: 'shcoeffs',
    18: 'shcoeffs',
    19: 'shcoeffs',
    20: 'shcoeffs',
    21: 'shcoeffs',
    22: 'shcoeffs',
    24: 'vae',
    25: 'vae',
    26: 'vae',
    27: 'vae',
    33: 'vae',
    35: 'vae',
    36: 'vae',
    37: 'vae'
}
CONDITIONS_CLASS = {
    0: AffineNoControl,
    1: AffineDepth,
    2: AffineNormal,
    3: AffineDepthNormal,
    4: AffineNoControl,
    5: AffineDepth,
    6: AffineNormal,
    7: AffineDepthNormal,
    8: AffineNormalBae,
    9: AffineDepthNormalBae,
    10: AffineNormalBae,
    11: AffineDepthNormalBae,
    12: AffineNoControl,
    13: AffineDepth,
    14: AffineNormal,
    15: AffineDepthNormal,
    16: AffineNoControl,
    17: AffineDepth,
    18: AffineNormal,
    19: AffineDepthNormal,
    20: AffineNormalBae,
    21: AffineDepthNormalBae,
    22: AffineNormalBae,
    23: AffineDepthNormalBae,
    24: AffineNoControl,
    25: AffineDepth,
    26: AffineDepthNormalBae,
    27: AffineNormalBae,
    33: AffineNoControl,
    35: AffineDepthNormalBae,
    36: AffineNormalBae,
    37: AffineDepth
}
LRS = {
    0: '1e-4',
    1: '1e-4',
    2: '1e-4',
    3: '1e-4',
    4: '1e-5',
    5: '1e-5',
    6: '1e-5',
    7: '1e-5',
    8: '1e-4',
    9: '1e-4',
    10: '1e-5',
    11: '1e-5',
    12: '1e-4',
    13: '1e-4',
    14: '1e-4',
    15: '1e-4',
    16: '1e-4',
    17: '1e-4',
    18: '1e-4',
    19: '1e-5',
    20: '1e-4',
    21: '1e-4',
    22: '1e-5',
    23: '1e-5',
    24: '1e-4',
    25: '1e-4',
    26: '1e-4',
    27: '1e-4',
    33: '1e-4',
    35: '1e-4',
    36: '1e-4',
    37: '1e-4'
}



def get_from_mode(mode):
    if mode == "ddim_left2right":
        return "/data/pakkapon/datasets/unsplash-lite/train_under", 1, DDIMUnsplashLiteDataset,{'index_file': 'src/20240824/ddim_left2right100.json'}, None
    elif mode == "ddim_left2right_dev":
        return "/data/pakkapon/datasets/unsplash-lite/train_under", 1, DDIMUnsplashLiteDataset,{'index_file': 'src/20240824/ddim_dev.json'}, None
    elif mode == "face10":
        return "datasets/face10", 1000, DDIMCrossDataset,{'index_file': None, 'envmap_file':'datasets/face10/target_envmap.json', 'envmap_dir':"/data/pakkapon/datasets/unsplash-lite/train_under" }, None
    elif mode == "shoe":
        return "/data/pakkapon/datasets/shoe_validation", 60, DDIMDataset, {'index_file': '/data/pakkapon/datasets/shoe_validation/ddim.json'}, None
    elif mode == "shoe_trainlight2":
        control_paths = {
            'control_depth': '/data/pakkapon/datasets/shoe_validation/control_depth/00000.png',
            'control_normal': '/data/pakkapon/datasets/shoe_validation/control_normal/00000.png', 
            'control_normal_bae': '/data/pakkapon/datasets/shoe_validation/control_normal_bae/00000.png',
        }
        source_env_ldr = '/data/pakkapon/datasets/shoe_validation/env_ldr/00000.png'
        source_env_under = '/data/pakkapon/datasets/shoe_validation/env_under/00000.png'
        image_path = '/data/pakkapon/datasets/shoe_validation/images/00000.png'
        return "/data/pakkapon/datasets/unsplash-lite/train_under", 10, DDIMSingleImageDataset, {'index_file': 'src/20240902/ddim_10right1left.json', 'image_path': image_path, 'control_paths': control_paths, 'source_env_ldr': source_env_ldr, 'source_env_under': source_env_under}, None
    elif mode == "face_left_ddim_v2":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-x-minus.json"}, None
    elif mode == "face_right_ddim_v2":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-x-plus.json"}, None
    elif mode == "face_identity_guidacne_hfcode":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-identity.json"}, None
    elif mode == "face_identity_guidacne_hfcode":
        return "/data/pakkapon/datasets/face/face2000", 100, DDIMDataset,{"index_file":"/data/pakkapon/datasets/face/face2000/split-identity.json"}, None
    elif mode == "multillum_train_v2":
        return "/data/pakkapon/datasets/multi_illumination/spherical/train", 100, DDIMSHCoeffsDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-train3scenes.json"}, None
    elif mode == "multillum_test_v2":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMSHCoeffsDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test3scenes.json"}, None
    elif mode == "multillum_val_array_v2":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-array.json"}, None   
    elif mode == "multillum_val":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val", 100, DDIMDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight.json"}, None   
    elif mode == "multillum_val_rotate_test":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/val_rotate/split.json"}, None   
    elif mode == "multillum_test_ddim30":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset, {"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test-ddim30.json"}, None   
    elif mode == "multillum_val_array_v4":
        return "/data/pakkapon/datasets/multi_illumination/spherical/val", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-array.json"}, None   
    elif mode == "multillum_test30_strength":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test-30-array.json"}, None
    elif mode == "multillum_test1_4light_strength":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-test-1-light-4-array.json"}, None
    elif mode == "multillum_val_strength":
        return "/data/pakkapon/datasets/multi_illumination/spherical/test", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-val-30-array.json"}, None
    elif mode == "multillum_train2_relight":
        return "/data/pakkapon/datasets/multi_illumination/spherical/train", 100, DDIMArrayEnvDataset,{"index_file":"/data/pakkapon/datasets/multi_illumination/spherical/split-train2-relight-array.json"}, None
    else:
        raise Exception("mode not found")

def main():
    CONDITIONS_CLASS[0] = AffineNoControl
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]


    strengthes = [float(a.strip()) for a in args.strength.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]

    for mode in modes:
        for version in versions:
            # parse checkpoint
            checkpoints = []
            for checkpoint_name in args.checkpoint.split(","):
                checkpoint_name = checkpoint_name.strip()
                if checkpoint_name == "lastest":
                    # find lastest checkpoint in given directory
                    checkpoint_dir = f"output/{CHECKPOINT_FOLDER_NAME}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints"
                    if not os.path.exists(checkpoint_dir):
                        print(f"Checkpoint dir not found: {checkpoint_dir}")
                        continue
                    checkpoint_lists = [int(a.split("=")[1].split(".")[0]) for a in os.listdir(checkpoint_dir) if a.startswith("epoch=")]
                    checkpoint_lists.sort()
                    lastest_checkpoint = checkpoint_lists[-1]
                    checkpoints.append(lastest_checkpoint)
                else:
                    checkpoints.append(int(checkpoint_name))    
                #condition_class = CONDITIONS_CLASS[version]
                #ddim_class = create_ddim_inversion(condition_class)
                ddim_class = CONDITIONS_CLASS[version]
                #try:
                if True:
                    for checkpoint in checkpoints:
                        if checkpoint == 0:
                            model = ddim_class(learning_rate=1e-4)
                            CKPT_PATH = None
                        else:
                            CKPT_PATH = f"output/{CHECKPOINT_FOLDER_NAME}/multi_mlp_fit/lightning_logs/version_{version}/checkpoints/epoch={checkpoint:06d}.ckpt"
                            if not os.path.exists(CKPT_PATH):
                                print(f"Checkpoint not found: {CKPT_PATH}")
                                continue
                            model = ddim_class.load_from_checkpoint(CKPT_PATH)
                        # disable chromeball inpaint if exist
                        if hasattr(model, 'pipe_chromeball'):
                            del model.pipe_chromeball
                        model.eval() # disable randomness, dropout, etc...
                        model.disable_plot_train_loss()
                        # set guidance bothway 
                        for guidance_scale in guidance_scales:
                            for strength in strengthes:
                                # compute inversion step
                                if strength > 1.0 or strength <= 0.0:
                                    print(f"Skip strength {strength} since it is not in range (0, 1]")
                                    continue
                                #inversion_step = np.clip(0,999, 1000 * strength)
                                inversion_step = 999
                                model.set_guidance_scale(guidance_scale)
                                model.set_ddim_guidance(guidance_scale)
                                model.set_inversion_step(inversion_step)    
                                model.set_ddim_strength(strength)
                                model.disable_null_text()                
                                output_dir = f"output/{FOLDER_NAME}/val_{mode}/{METHODS[version]}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/strength{strength}"
                                # skip if output dir exist 
                                if os.path.exists(output_dir):
                                    print(f"Skip {output_dir}")
                                    continue
                                os.makedirs(output_dir, exist_ok=True)
                                print("================================")
                                print(output_dir)
                                print("================================")
                                trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=output_dir, inference_mode=False, gradient_clip_val=0)
                                val_root, count_file, dataset_class, dataset_args, specific_prompt = get_from_mode(mode)
                                if type(count_file) == int:
                                    split = slice(0, count_file, 1)
                                else:
                                    split = count_file
                                val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                                trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
                        continue
                            
                #except:
                #   pass

                                
if __name__ == "__main__":
    main()