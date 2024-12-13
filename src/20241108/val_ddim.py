# val_grid is a validation at step 
# bin/siatv100 src/20241108/val_ddim.py -i 90499
# bin/siatv100 src/20241108/val_ddim.py -i 90500
# bin/siatv100 src/20241108/val_ddim.py -i 90501
# bin/siatv100 src/20241108/val_ddim.py -i 90502
# bin/siatv100 src/20241108/val_ddim.py -i 90532
# bin/siatv100 src/20241108/val_ddim.py -i 90533
# bin/siatv100 src/20241108/val_ddim.py -i 90535
# bin/siatv100 src/20241108/val_ddim.py -i 90536
# bin/siatv100 src/20241108/val_ddim.py -i 90499 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90500 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90501 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90502 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90532 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90533 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90535 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90536 -m train2left,train2right
# bin/siatv100 src/20241108/val_ddim.py -i 90499 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90500 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90501 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90502 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90532 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90533 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90535 -m viz
# bin/siatv100 src/20241108/val_ddim.py -i 90536 -m viz
# validation 
# bin/siatv100 src/20241108/val_ddim.py -i 91542 -m valid2left
# bin/siatv100 src/20241108/val_ddim.py -i 91542 -m valid2right
# bin/siatv100 src/20241108/val_ddim.py -i 91542 -m train2left
# bin/siatv100 src/20241108/val_ddim.py -i 91542 -m train2right




# testrun
# bin/siatv100 src/20241108/val_ddim.py -i 90499 -m train_left_v2


# 90532: 'default',
# 90533: 'default',
# 90535: 'default',
# 90536: 'default'

import os 

import lightning as L
import torch
import argparse 
#from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sddiffusionface import SDDiffusionFace, ScrathSDDiffusionFace, SDWithoutAdagnDiffusionFace, SDOnlyAdagnDiffusionFace, SDOnlyShading, SDDiffusionFaceNoShading

from datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset

MASTER_TYPE = 16
CHECKPOINT_FOLDER_NAME = "20241108"


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
    89738: 'mint_pretrain',
    89740: 'mint_pretrain',
    90499: 'mint_pretrain', #all 1e-4
    90500: 'mint_pretrain', #all 1e-5
    90501: 'mint_scrath',
    90502: 'mint_scrath',
    90532: 'controlnet_only',
    90533: 'controlnet_only',
    90535: 'adagan_only',
    90536: 'adagan_only',
    91539: 'mint_pretrain', #all 1e-4
    91542: 'mint_pretrain', #all 1e-5
    91864: 'adagan_shcoeff',
    91865: 'adagan_shcoeff',
    91866: 'adagan_shcoeff',
    91869: 'clip_shcoeff', 
    91870: 'clip_shcoeff',
    91871: 'clip_shcoeff',
    91872: 'clip',
    91875: 'clip',
    91876: 'clip',
    92037: 'shcoeff_order2',
    92047: 'shcoeff_order2',
    92049: 'shcoeff_order2',
    92205: 'shading_control_only',
    92206: 'shading_control_only',
    92207: 'shading_control_only',
    92372: 'inpaint',
    92414: 'inpaint',
    92423: 'inpaint_only_background',
    92438: 'inpaint_only_background',
    92829: 'v2_defareli',
    92830: 'v2_defareli',
    92824: 'v2_adagn_face_shcoeff',
    92825: 'v2_adagn_face_shcoeff',
    92826: 'v2_adagn_only_shcoeff',
    92833: 'v2_adagn_only_shcoeff',
    93026: 'v2_defareli',
    93027: 'v2_adagn_only_shcoeff'
}
METHODS = {
    89738: 'default',
    89740: 'default',
    90499: 'default',
    90500: 'default',
    90501: 'default',
    90502: 'default',
    90532: 'default',
    90533: 'default',
    90535: 'default',
    90536: 'default',
    91539: 'default',
    91542: 'default',
    91864: 'default',
    91865: 'default',
    91866: 'default',
    91869: 'default',
    91870: 'default',
    91871: 'default',
    91872: 'default',
    91875: 'default',
    91876: 'default',
    92037: 'default',
    92047: 'default',
    92049: 'default',
    92205: 'default',
    92206: 'default',
    92207: 'default',
    92372: 'default',
    92414: 'default',
    92423: 'default',
    92438: 'default',
    92829: 'default',
    92830: 'default',
    92824: 'default',
    92825: 'default',
    92826: 'default',
    92833: 'default',
    93026: 'default',
    93027: 'default'
}
CONDITIONS_CLASS = {
    89738: SDDiffusionFace,
    89740: SDDiffusionFace,
    90499: SDDiffusionFace,
    90500: SDDiffusionFace,
    90501: ScrathSDDiffusionFace,
    90502: ScrathSDDiffusionFace,
    90532: SDWithoutAdagnDiffusionFace,
    90533: SDWithoutAdagnDiffusionFace,
    90535: SDOnlyAdagnDiffusionFace,
    90536: SDOnlyAdagnDiffusionFace,
    91539: SDDiffusionFace,
    91542: SDDiffusionFace,
    91864: SDOnlyAdagnDiffusionFace,
    91865: SDOnlyAdagnDiffusionFace,
    91866: SDOnlyAdagnDiffusionFace,
    91869: SDOnlyAdagnDiffusionFace,
    91870: SDOnlyAdagnDiffusionFace,
    91871: SDOnlyAdagnDiffusionFace,    
    91872: SDDiffusionFace,
    91875: SDDiffusionFace,
    91876: SDDiffusionFace,
    92037: SDOnlyAdagnDiffusionFace,
    92047: SDOnlyAdagnDiffusionFace,
    92049: SDOnlyAdagnDiffusionFace,
    92205: SDOnlyShading,
    92206: SDOnlyShading,
    92207: SDOnlyShading,
    92372: SDDiffusionFace,
    92414: SDDiffusionFace,
    92423: SDDiffusionFaceNoShading,
    92438: SDDiffusionFaceNoShading,
    92829: SDDiffusionFace,
    92830: SDDiffusionFace,
    92824: SDOnlyAdagnDiffusionFace,
    92825: SDOnlyAdagnDiffusionFace,
    92826: SDOnlyAdagnDiffusionFace,
    92833: SDOnlyAdagnDiffusionFace,
    93026: SDDiffusionFace,
    93027: SDOnlyAdagnDiffusionFace
}
LRS = {
    89738: '1e-4',
    89740: '1e-5',
    90499: '1e-4',
    90500: '1e-5',
    90501: '1e-5',
    90502: '5e-6',
    90532: '1e-4',
    90533: '1e-5',
    90535: '1e-4',
    90536: '1e-5',
    91539: '1e-4',
    91542: '1e-5',
    91864: '1e-4',
    91865: '1e-5',
    91866: '1e-6',
    91869: '1e-4',
    91870: '1e-5',
    91871: '1e-6',
    91872: '1e-4',
    91875: '1e-5',
    91876: '1e-6',
    92037: '1e-4',
    92047: '1e-5',
    92049: '1e-6',
    92205: '1e-4',
    92206: '1e-5',
    92207: '1e-6',
    92372: '1e-4',
    92414: '1e-5',
    92423: '1e-4',
    92438: '1e-5', 
    92829: '1e-4',
    92830: '1e-5',
    92824: '1e-4',
    92825: '1e-5',
    92826: '1e-4',
    92833: '1e-5',
    93026: '1e-5',
    93027: '1e-5'
}
DIRNAME = {
    89738: CHECKPOINT_FOLDER_NAME,
    89740: CHECKPOINT_FOLDER_NAME,
    90499: CHECKPOINT_FOLDER_NAME,
    90500: CHECKPOINT_FOLDER_NAME,
    90501: CHECKPOINT_FOLDER_NAME,
    90502: CHECKPOINT_FOLDER_NAME,
    90532: CHECKPOINT_FOLDER_NAME,
    90533: CHECKPOINT_FOLDER_NAME,
    90535: CHECKPOINT_FOLDER_NAME,
    90536: CHECKPOINT_FOLDER_NAME,
    91539: CHECKPOINT_FOLDER_NAME,
    91542: CHECKPOINT_FOLDER_NAME,
    91864: CHECKPOINT_FOLDER_NAME,
    91865: CHECKPOINT_FOLDER_NAME,
    91866: CHECKPOINT_FOLDER_NAME,
    91869: CHECKPOINT_FOLDER_NAME,
    91870: CHECKPOINT_FOLDER_NAME,
    91871: CHECKPOINT_FOLDER_NAME,
    91872: CHECKPOINT_FOLDER_NAME,
    91875: CHECKPOINT_FOLDER_NAME,
    91876: CHECKPOINT_FOLDER_NAME,
    92037: CHECKPOINT_FOLDER_NAME,
    92047: CHECKPOINT_FOLDER_NAME,
    92049: CHECKPOINT_FOLDER_NAME,
    92205: CHECKPOINT_FOLDER_NAME,
    92206: CHECKPOINT_FOLDER_NAME,
    92207: CHECKPOINT_FOLDER_NAME,
    92372: CHECKPOINT_FOLDER_NAME,
    92414: CHECKPOINT_FOLDER_NAME,
    92423: CHECKPOINT_FOLDER_NAME,
    92438: CHECKPOINT_FOLDER_NAME,
    92829: CHECKPOINT_FOLDER_NAME,
    92830: CHECKPOINT_FOLDER_NAME,
    92824: CHECKPOINT_FOLDER_NAME,
    92825: CHECKPOINT_FOLDER_NAME,
    92826: CHECKPOINT_FOLDER_NAME,
    92833: CHECKPOINT_FOLDER_NAME,
    93026: CHECKPOINT_FOLDER_NAME,
    93027: CHECKPOINT_FOLDER_NAME
}
CHECKPOINTS = {
    89738: 24,
    89740: 24,
    90499: 42,
    90500: 43,
    90501: 34,
    90502: 34,
    90532: 8,
    90533: 8,
    90535: 11,
    90536: 10,
    91539: 50,
    91542: 50,
    91864: 33,
    91865: 33,
    91866: 33, 
    91869: 24,
    91870: 24,
    91871: 24, 
    91872: 18,
    91875: 18,
    91876: 18, 
    92037: 33,
    92047: 33,
    92049: 33, 
    92205: 9,
    92206: 9,
    92207: 9,
    92372: 6,
    92414: 6,
    92423: 6,
    92438: 6,
    92829: 9,
    92830: 9,
    92824: 9,
    92825: 9,
    92826: 9,
    92833: 9,
    93026: 10,
    93027: 10
}

use_shcoeff2 = [91864, 91865, 91866, 91869, 91870, 91871, 92037, 92047, 92049, 92824, 92825, 92826, 92833, 93027]
use_only_light = [92037, 92047, 92049, 92826, 92833, 93027]
use_random_mask_background = [92372, 92414, 92423, 92438]

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
    elif mode == "valid_spatial":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json"}, "a photorealistic image"
    elif mode == "valid_spatial_test":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test/index-array.json"}, "a photorealistic image"
    elif mode == "valid_spatial_test2":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test2", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test2/index-array.json"}, "a photorealistic image"
    elif mode == "valid_spatial_test3":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test2", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test3/index-array.json"}, "a photorealistic image"
    elif mode == "valid_spatial_test4":
        return "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test2", 100, DDIMDiffusionFaceRelightDataset,{"index_file":"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial_test4/index-array.json"}, "a photorealistic image"
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
                            val_dataset = dataset_class(split=split, root_dir=val_root, specific_prompt=specific_prompt, **dataset_args)
                            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)


                except Exception as e:
                    raise e

                                
if __name__ == "__main__":
    main()