import argparse 
from datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset
from SDRelightShading import SDRelightShading
from constants import FOLDER_NAME
import os
import torch
import torchvision 
import numpy as np
MASTER_TYPE = torch.float16
PRESET = {
    'batch_8': [],
}

# define checkpoint for batch 8
for chk  in range(260000, 0, -20000):
    PRESET['batch_8'].append({
        'name': f'chk{chk}',
        'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-5/batch_8/checkpoint-{chk}/controlnet'
    })

def get_builder_from_mode(mode):
    if mode == "rotate_everett_dining1":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes",
                "index_file":"/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes/everett_dining1_rotate.json",
                "shadings_dir": "control_shading_from_fitting_v3_exr",
                "backgrounds_dir": 'control_shading_from_fitting_v3_exr',
                "images_dir":"control_render_from_fitting_v2", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "everett_kitchen2_rotate":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes",
                "index_file":"/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes/everett_kitchen2_rotate.json",
                "shadings_dir": "control_shading_from_fitting_v3_exr",
                "backgrounds_dir": 'control_shading_from_fitting_v3_exr',
                "images_dir":"control_render_from_fitting_v2", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "everett_kitchen4_rotate":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes",
                "index_file":"/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes/everett_kitchen4_rotate.json",
                "shadings_dir": "control_shading_from_fitting_v3_exr",
                "backgrounds_dir": 'control_shading_from_fitting_v3_exr',
                "images_dir":"control_render_from_fitting_v2", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "everett_kitchen6_rotate":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes",
                "index_file":"/data/pakkapon/datasets/multi_illumination/spherical/val_rotate_test_scenes/everett_kitchen6_rotate.json",
                "shadings_dir": "control_shading_from_fitting_v3_exr",
                "backgrounds_dir": 'control_shading_from_fitting_v3_exr',
                "images_dir":"control_render_from_fitting_v2", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    raise Exception('Please select correct dataset')

@torch.inference_mode()
def main(args):
    checkpoints = []
    if args.preset != '':
        checkpoints = PRESET[args.preset]
    else:
        new_checkpoints = args.checkpoint.split(',')
        for idx, n in enumerate(new_checkpoints):
            checkpoints.append({
                'name': f"{idx}",
                'path': n
            })
    #versions = [int(a.strip()) for a in args.version.split(",")]

    # load pipe 
    for meta in checkpoints:
        # load dataset 
        dataset_builder = get_builder_from_mode(args.mode)
        val_dataset = dataset_builder['dataset_class'](**dataset_builder['dataset_params'])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        output_dir = f"output/{FOLDER_NAME}/val_{args.mode}/{meta['name']}/"
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)
        crop_dir = os.path.join(output_dir, 'crop_image')
        os.makedirs(crop_dir, exist_ok=True)
        os.chmod(crop_dir, 0o777)

        with_groudtruth_dir = os.path.join(output_dir, 'with_groudtruth')
        os.makedirs(with_groudtruth_dir, exist_ok=True)
        os.chmod(with_groudtruth_dir, 0o777)

        control_dir = os.path.join(output_dir, 'control')
        os.makedirs(control_dir, exist_ok=True)
        os.chmod(control_dir, 0o777)

        pipe = SDRelightShading(
            controlnet_path = meta['path']
        )
        device = pipe.pipe.device
        for batch in val_dataloader:
            init_latent = None
            for target_idx in range(len(batch['target_shading'])):
                print("SOURCE: ", batch['name'][0], " | Target: ", batch['word_name'][target_idx][0])

                ret = pipe.relight(
                    source_image = batch['source_image'].to(device).to(MASTER_TYPE),
                    target_shading = batch['target_shading'][target_idx].to(device).to(MASTER_TYPE),
                    prompt=batch['text'][0],
                    source_shading = batch['source_shading'].to(device).to(MASTER_TYPE),
                    latents = init_latent
                )
                init_latent = ret['init_latent']
                image = ret['image']

                # save image
                image = torch.clamp(image, 0.0, 1.0).cpu()
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                torchvision.utils.save_image(image, os.path.join(crop_dir, f"{filename}.png"))
                os.chmod(os.path.join(crop_dir, f"{filename}.png"), 0o777)
                
                control_image = batch['target_shading'][target_idx]
                control_image = control_image / control_image.max()
                filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                torchvision.utils.save_image(control_image, os.path.join(control_dir, f"{filename}.png"))
                os.chmod(os.path.join(control_dir, f"{filename}.png"), 0o777)

                gt_image = (batch['target_image'][target_idx] + 1.0) / 2.0
                gt_image = gt_image.to(image.device)
                tb_image = [gt_image, image, control_image]
                tb_image = torch.cat(tb_image, dim=0)
                tb_image = torch.clamp(tb_image, 0.0, 1.0)
                tb_image = torchvision.utils.make_grid(tb_image, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
                torchvision.utils.save_image(tb_image, os.path.join(with_groudtruth_dir, f"{filename}.jpg"))
                os.chmod(os.path.join(with_groudtruth_dir, f"{filename}.jpg"), 0o777)


        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--version", type=str, default="2")
    parser.add_argument("-m", "--mode", type=str, default="rotate_everett_dining1")
    parser.add_argument("-c", "--checkpoint", type=str, default="")
    parser.add_argument("-p", "--preset", type=str, default="batch_8")
    args = parser.parse_args()
    main(args)

"""
key avalible in batch
name
source_image
text
word_name
idx
source_diffusion_face
source_background
source_shading
target_image
target_diffusion_face
target_background
target_shading
"""