import argparse 
from vll_datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset
from SDRelightShading import SDRelightShading
from constants import FOLDER_NAME
import os
import torch
import torchvision 
import torchmetrics
import numpy as np
import json
MASTER_TYPE = torch.float16
PRESET = {
    'batch_8_run1': [
        {
        'name': f'chk24',
        'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8/checkpoint-80000/controlnet'
        }
    ],
    'batch_8_run3': [
        {
        'name': f'chk79',
        'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v3/checkpoint-246320/controlnet'
        }
    ],
    'full1e-5':[
        {
        'name': f'checkpoint-80000',
        'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/controlnet/v1/1e-5/checkpoint-80000/controlnet'
        }
    ],
    'full1e-4':[
        {
        'name': f'checkpoint-180000',
        'path': f'/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/v1_old/1e-4/checkpoint-180000/controlnet'
        }
    ],
    'full1e-4_v2':[
        {
        'name': f'checkpoint-120000',
        'path': f'/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/v1_old/1e-4/checkpoint-120000/controlnet'
        }
    ],
    'seed_rotate_run1':[
        # {
        #     'name': 'run_01_chk80',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-246320/controlnet'
        # },
        # {
        #     'name': 'run_01_chk60',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-184740/controlnet'
        # },
        {
            'name': 'run_01_chk40',
            'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-123160/controlnet'
        },
    ],
    'seed_rotate_run2':[
        # {
        #     'name': 'run_02_chk80',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v2/checkpoint-246320/controlnet'
        # },
        # {
        #     'name': 'run_02_chk60',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v2/checkpoint-184740/controlnet'
        # },
        # {
        #     'name': 'run_02_chk40',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v2/checkpoint-123160/controlnet'
        # }
    ],
    'seed_rotate_run3':[
        # {
        #     'name': 'run_03_chk80',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v3/checkpoint-246320/controlnet'
        # },
        # {
        #     'name': 'run_03_chk60',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v3/checkpoint-184740/controlnet'
        # },
        {
            'name': 'run_03_chk40',
            'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v3/checkpoint-123160/controlnet'
        },
    ],
    'seed_rotate_run4':[
        # {
        #     'name': 'run_04_chk35',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v4/checkpoint-107765/controlnet'
        # }
        {
            'name': 'run_04_chk40',
            'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v4/checkpoint-123160/controlnet'
        }
    ],
    'seed_rotate_run5':[
        # {
        #     'name': 'run_05_chk35',
        #     'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v5/checkpoint-107765/controlnet'
        # }
        {
            'name': 'run_05_chk35',
            'path': '/pure/t1/checkpoints/sd-light-time/20250425_huggingface_controlnet/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v5/checkpoint-123160/controlnet'
        }
    ],
    'batch_8_v1_r1': [
        {
            'name': f'chk80',
            'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/output_t1/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-246320/controlnet'
        },
        # {
        #     'name': f'chk20',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/output_t1/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-61580/controlnet'
        # },
        # {
        #     'name': f'chk15',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/output_t1/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-46185/controlnet'
        # },
        # {
        #     'name': f'chk10',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/output_t1/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-30790/controlnet'
        # },
        # {
        #     'name': f'chk5',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/output_t1/controlnet/MultiLstShading/v1/learning_rate_1e-4/batch_8_v1/checkpoint-15395/controlnet'
        # }
    ],
    'code0510_least_square_1e-4': [
        {
            'name': f'chk80',
            'path': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v3/1e-4/checkpoint-246320/controlnet',
        },
        # {
        #     'name': f'chk60',
        #     'path': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v3/1e-4/checkpoint-184740/controlnet',
        # },
        # {
        #     'name': f'chk40',
        #     'path': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v3/1e-4/checkpoint-123160/controlnet',
        # },
        
    ],
    'code0510_least_square_1e-5': [
        {
            'name': f'chk80',
            'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v2/1e-5/checkpoint-246320/controlnet'
        },
        # {
        #     'name': f'chk60',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v2/1e-5/checkpoint-184740/controlnet'
        # },
        # {
        #     'name': f'chk40',
        #     'path': f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_least_square_diffusionlight/v2/1e-5/checkpoint-123160/controlnet'
        # },
       
    ],
}




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
    if mode == "rotate_everett_kitchen2":
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
    if mode == "rotate_everett_kitchen4":
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
    if mode == "rotate_everett_kitchen6":
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
    
    if mode == "rotate_diffusionlight_everett_dining1":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate",
                "index_file":"/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate/index/everett_dining1_rotate.json",
                "shadings_dir": "shadings_marigold_v2",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_diffusionlight_everett_kitchen2":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate",
                "index_file":"/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate/index/everett_kitchen2_rotate.json",
                "shadings_dir": "shadings_marigold_v2",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_diffusionlight_everett_kitchen4":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate",
                "index_file":"/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate/index/everett_kitchen4_rotate.json",
                "shadings_dir": "shadings_marigold_v2",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_diffusionlight_everett_kitchen6":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate",
                "index_file":"/data/pakkapon/datasets/multi_illumination/shadings/least_square/v3/rotate/index/everett_kitchen6_rotate.json",
                "shadings_dir": "shadings_marigold_v2",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    raise Exception('Please select correct dataset')

@torch.inference_mode()
def main(args):
    _ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure
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
    modes = args.mode.split(',')
    seeds = [int(s) for s in args.seed.split(',')]
    # load pipe 
    for mode in modes:
        for meta in checkpoints:
            for seed in seeds:
                # load dataset 
                dataset_builder = get_builder_from_mode(mode)
                val_dataset = dataset_builder['dataset_class'](**dataset_builder['dataset_params'])
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
                experiment_name = args.preset if not 'experiment' in meta else meta['experiment']
                output_dir = f"../../output/{FOLDER_NAME}/val_{mode}/{experiment_name}/{meta['name']}/seed{seed}"
                
                os.makedirs(output_dir, exist_ok=True)
                os.chmod(f"../../output/{FOLDER_NAME}/val_{mode}/{experiment_name}/{meta['name']}", 0o777)
                os.chmod(f"../../output/{FOLDER_NAME}/val_{mode}/{experiment_name}", 0o777)
                os.chmod(f"../../output/{FOLDER_NAME}/val_{mode}", 0o777)
                crop_dir = os.path.join(output_dir, 'crop_image')
                os.makedirs(crop_dir, exist_ok=True)
                os.chmod(crop_dir, 0o777)
                print("SUCESS MMAKE")

                with_groudtruth_dir = os.path.join(output_dir, 'with_groudtruth')
                os.makedirs(with_groudtruth_dir, exist_ok=True)
                os.chmod(with_groudtruth_dir, 0o777)

                control_dir = os.path.join(output_dir, 'control')
                os.makedirs(control_dir, exist_ok=True)
                os.chmod(control_dir, 0o777)

                mse_dir = os.path.join(output_dir, 'mse')
                os.makedirs(mse_dir, exist_ok=True)
                os.chmod(mse_dir, 0o777)

                psnr_dir = os.path.join(output_dir, 'psnr')
                os.makedirs(psnr_dir, exist_ok=True)
                os.chmod(psnr_dir, 0o777)

                ssim_dir = os.path.join(output_dir, 'ssim')
                os.makedirs(ssim_dir, exist_ok=True)
                os.chmod(ssim_dir, 0o777)

                ddsim_dir = os.path.join(output_dir, 'ddsim')
                os.makedirs(ddsim_dir, exist_ok=True)
                os.chmod(ddsim_dir, 0o777)

                pipe = SDRelightShading(
                    controlnet_path = meta['path'],
                    seed = seed
                )
                device = pipe.pipe.device
                scores = {
                    'psnr': [],
                    'mse': []
                }
                for batch in val_dataloader:
                    init_latent = None
                    for target_idx in range(len(batch['target_shading'])):
                        print("SOURCE: ", batch['name'][0], " | Target: ", batch['word_name'][target_idx][0])
                        filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                        with_gt_path = os.path.join(with_groudtruth_dir, f"{filename}.jpg")
                        if os.path.exists(with_gt_path):
                            print("Already exists")
                            continue
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
                        
                        torchvision.utils.save_image(image, os.path.join(crop_dir, f"{filename}.png"))
                        os.chmod(os.path.join(crop_dir, f"{filename}.png"), 0o777)
                        
                        control_image = batch['target_shading'][target_idx]
                        control_image = control_image / control_image.max()
                        
                        torchvision.utils.save_image(control_image, os.path.join(control_dir, f"{filename}.png"))
                        os.chmod(os.path.join(control_dir, f"{filename}.png"), 0o777)
    
                        gt_image = (batch['target_image'][target_idx] + 1.0) / 2.0
                        gt_image = gt_image.to(image.device)
                        
                        mse = torch.nn.functional.mse_loss(gt_image, image, reduction="none").mean().cpu()
                        psnr = -10 * torch.log10(mse)
                        #ssim = _ssim_loss(gt_image, image)
                        #ddsim = (1.0 - ssim) / 2.0

                        with open(f"{mse_dir}/{filename}.txt", "w") as f:
                            f.write(f"{mse.item()}\n")
                        with open(f"{psnr_dir}/{filename}.txt", "w") as f:
                            f.write(f"{psnr.item()}\n")
                        scores['psnr'].append(psnr.item())
                        scores['mse'].append(mse.item())

                        tb_image = [gt_image, image, control_image]
                        tb_image = torch.cat(tb_image, dim=0)
                        tb_image = torch.clamp(tb_image, 0.0, 1.0)
                        tb_image = torchvision.utils.make_grid(tb_image, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
                        torchvision.utils.save_image(tb_image, os.path.join(with_groudtruth_dir, f"{filename}.jpg"))
                        os.chmod(with_gt_path, 0o777)


                # finding average for saving
                for k in ['psnr','mse']:
                    scores[k] = np.mean(np.array(scores[k])).item()

                with open(f"{output_dir}/scores.json", "w") as f:
                    json.dump(scores, f, indent=4)





        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--version", type=str, default="2")
    #parser.add_argument("-m", "--mode", type=str, default="rotate_everett_kitchen6,rotate_everett_dining1,rotate_everett_kitchen2,rotate_everett_kitchen4")
    parser.add_argument("-c", "--checkpoint", type=str, default="")
    #parser.add_argument("-p", "--preset", type=str, default="batch_8_run1")
    parser.add_argument("-p", "--preset", type=str, default="rotate_3_runs")
    parser.add_argument("-m", "--mode", type=str, default="rotate_everett_kitchen6,rotate_everett_dining1,rotate_everett_kitchen2,rotate_everett_kitchen4")
    parser.add_argument('-seed', type=str, default='42')
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