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
OUTPUT_DIR = "../../output_t1"
PRESET = {
    # '1e-4_lstsq_image_lstsq_shading': [
    #     {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-4/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-4/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-4/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-5_lstsq_image_lstsq_shading': [
    #     {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-5/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-5/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-5/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-4_real_image_lstsq_shading': [
    #     {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_lstsq_shading/v1/1e-4/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-4/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-4/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-5_real_image_lstsq_shading': [
    #      {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_lstsq_shading/v1/1e-5/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-5/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1/1e-5/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-4_lstsq_image_real_shading': [
    #     {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-5_lstsq_image_real_shading': [
    #     {
    #         'name': f'chk30',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-92370/controlnet"
    #     },
    #     {
    #         'name': f'chk20',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-61580/controlnet"
    #     },
    #     {
    #         'name': f'chk10',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/1e-4/checkpoint-30790/controlnet"
    #     }
    # ],
    # '1e-4_real_image_real_shading': [
    #     {
    #         'name': f'checkpoint-13000',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/shading_webdataset_maxshading32_batch32/v1/1e-4/checkpoint-16000/controlnet"
    #     }
    # ],
    # '1e-5_real_image_real_shading': [
    #     {
    #         'name': f'checkpoint-13000',
    #         'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/shading_webdataset_maxshading32_batch32/v1/1e-5/checkpoint-17000/controlnet"
    #     }
    # ],
}

checkpoint_dir ={
    'lstsq_image_lstsq_shading': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_lstsq_image_lstsq_shading/v1',
    'lstsq_image_real_shading': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4',
    'real_image_lstsq_shading': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_lstsq_shading/v1',
    'real_image_real_shading': '/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250510_webdataset_support/controlnet/multi_illumination_real_diffusionlight/v4'
}

for image_type in ['real','lstsq']:
    for shading_type in ['real','lstsq']:
        for lr in ['1e-4', '1e-5']:
            exp_type = f'{image_type}_image_{shading_type}_shading'
            name = f"{lr}_{exp_type}"
            PRESET[name] = []
            for chk in [60]:
                PRESET[name].append({
                    'name': f'chk{chk}',
                    'path': os.path.join(checkpoint_dir[exp_type], lr, f'checkpoint-{chk * 3079}', 'controlnet')
                })




# multi_illumination_vary_scene_985 (using lstsq shading and lstsq image)
PRESET['multi_illumination_vary_scene_985'] = []
for checkpoint in [110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]:
    PRESET['multi_illumination_vary_scene_985'].append({
        'name': f'chk{checkpoint}',
        'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illuination_vary_scene/v1/num_scene_985/checkpoint-{checkpoint * 3079}/controlnet"
    })

for dataset_size in [500, 250, 100, 50, 20, 10, 5]:
    PRESET[f'multi_illumination_vary_scene_{dataset_size}'] = []
    for checkpoint in [110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]:
        data = {
            'name': f'chk{checkpoint}',
            'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illuination_vary_scene/v1/num_scene_{dataset_size:03d}/checkpoint-{checkpoint * 3125}/controlnet"
        }
        if os.path.exists(data['path']):
            PRESET[f'multi_illumination_vary_scene_{dataset_size}'].append(data)



PRESET['multi_illumination_real_image_lstsq_shading_v0_hf'] = []
for checkpoint in [110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2,1]:
    PRESET['multi_illumination_real_image_lstsq_shading_v0_hf'].append({
        'name': f'chk{checkpoint}',
        'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_lstsq_shading/v0_hf/1e-4/checkpoint-{checkpoint * 3079}/controlnet"
    })

PRESET['multi_illumination_real_image_real_shading_v0_hf'] = []
for checkpoint in [80]:
    PRESET['multi_illumination_real_image_real_shading_v0_hf'].append({
        'name': f'chk{checkpoint}',
        'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_real_shading/v0_hf/1e-4/checkpoint-{checkpoint * 3079}/controlnet"
    })

PRESET['multi_illumination_real_image_real_shading_v0_hf_max32_1e-4'] = []
for checkpoint in  [110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2,1]: #  110, 100, 90, 60, 40, 20, 100, 90, 70, 50, 30, 10
    PRESET['multi_illumination_real_image_real_shading_v0_hf_max32_1e-4'].append({
        'name': f'chk{checkpoint}',
        'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_real_shading/v0_hf_max32/1e-4/checkpoint-{checkpoint * 3079}/controlnet"
    })


PRESET['multi_illumination_real_image_real_shading_v0_hf_max32_1e-5'] = []
for checkpoint in [80]:
    PRESET['multi_illumination_real_image_real_shading_v0_hf_max32_1e-5'].append({
        'name': f'chk{checkpoint}',
        'path': f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/controlnet/multi_illumination_real_image_real_shading/v0_hf_max32/1e-5/checkpoint-{checkpoint * 3079}/controlnet"
    })

max_shadings = ['multi_illumination_real_image_real_shading_v0_hf_max32_1e-4', 'multi_illumination_real_image_real_shading_v0_hf_max32_1e-5']
            
def get_builder_from_mode(mode):
    if mode == "training_set":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/train",
                "index_file":"/pure/t1/datasets/laion-shading/v4/train/index/training_set.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "everett_dining1_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/test",
                "index_file":"/pure/t1/datasets/laion-shading/v4/test/index/everett_dining1.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "14n_copyroom_1_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train/index/14n_copyroom1.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "14n_copyroom_6_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train/index/14n_copyroom6.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_8_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train/index/14n_copyroom8.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_10_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_real_shading/v0/train/index/14n_copyroom10.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "rotate_everett_dining1_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/rotate",
                "index_file":"/pure/t1/datasets/laion-shading/v4/rotate/index/everett_dining1.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_everett_kitchen2_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/rotate",
                "index_file":"/pure/t1/datasets/laion-shading/v4/rotate/index/everett_kitchen2.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_everett_kitchen4_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/rotate",
                "index_file":"/pure/t1/datasets/laion-shading/v4/rotate/index/everett_kitchen4.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "rotate_everett_kitchen6_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/rotate",
                "index_file":"/pure/t1/datasets/laion-shading/v4/rotate/index/everett_kitchen6.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "14n_copyroom_1_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train/index/14n_copyroom1.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "14n_copyroom_6_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train/index/14n_copyroom6.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_8_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train/index/14n_copyroom8.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_10_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/real_image_lstsq_shading/v0/train/index/14n_copyroom10.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }


    if mode == "14n_copyroom_1_least_square_image_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train/index/14n_copyroom1.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "14n_copyroom_6_least_square_image_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train/index/14n_copyroom6.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_8_least_square_image_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train/index/14n_copyroom8.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    
    if mode == "14n_copyroom_10_least_square_image_least_square_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train",
                "index_file":"/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/train/index/14n_copyroom10.json",
                "shadings_dir": "shadings",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }



    if mode == "everett_kitchen2_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/test",
                "index_file":"/pure/t1/datasets/laion-shading/v4/test/index/everett_kitchen2.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
    if mode == "everett_kitchen4_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/test",
                "index_file":"/pure/t1/datasets/laion-shading/v4/test/index/everett_kitchen4.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }
        
    if mode == "everett_kitchen6_diffusionlight_shading":
        return {
            'split': 100,
            'dataset_params': {
                'root_dir': "/pure/t1/datasets/laion-shading/v4/test",
                "index_file":"/pure/t1/datasets/laion-shading/v4/test/index/everett_kitchen6.json",
                "shadings_dir": "shadings_marigold",
                "backgrounds_dir": 'images',
                "images_dir":"images", 
                'specific_prompt': None,
                'feature_types': []
            }, 
            'dataset_class': DDIMDiffusionFaceRelightDataset
        }

    if mode == "rotate_everett_dining1_least_square_shading":
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

    if mode == "rotate_everett_dining1_least_square_shading":
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
    if mode == "rotate_everett_kitchen2_least_square_shading":
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
    if mode == "rotate_everett_kitchen4_least_square_shading":
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
    if mode == "rotate_everett_kitchen6_least_square_shading":
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

def normalize_with_max_shading(image, max_shading):
    image = image / max_shading # range [0, 1]
    image = image * 2 - 1 # range [-1, 1]
    return image

@torch.inference_mode()
def main(args):
    _ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure
    presets = args.preset.split(',')
   
    for preset in presets:    
        checkpoints = []
        if preset != '':
            checkpoints = PRESET[preset]
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
                    experiment_name = preset if not 'experiment' in meta else meta['experiment']
                    output_dir = f"{OUTPUT_DIR}/{FOLDER_NAME}/val_{mode}/{experiment_name}/{meta['name']}/seed{seed}"

                    print(f"Output directory: {output_dir}")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    os.chmod(output_dir, 0o777)
                    try:
                        os.chmod(f"{OUTPUT_DIR}/{FOLDER_NAME}/val_{mode}/{experiment_name}/{meta['name']}", 0o777)
                        os.chmod(f"{OUTPUT_DIR}/{FOLDER_NAME}/val_{mode}/{experiment_name}", 0o777)
                        os.chmod(f"{OUTPUT_DIR}/{FOLDER_NAME}/val_{mode}", 0o777)
                    except:
                        pass

                    crop_dir = os.path.join(output_dir, 'crop_image')
                    os.makedirs(crop_dir, exist_ok=True)
                    os.chmod(crop_dir, 0o777)

                    with_groudtruth_dir = os.path.join(output_dir, 'with_groudtruth')
                    os.makedirs(with_groudtruth_dir, exist_ok=True)
                    os.chmod(with_groudtruth_dir, 0o777)

                    sdoutput_dir = os.path.join(output_dir, 'sd_output')
                    os.makedirs(sdoutput_dir, exist_ok=True)
                    os.chmod(sdoutput_dir, 0o777)


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
                    
                    if not os.path.exists(meta['path']):
                        print(f"Checkpoint {meta['path']} does not exist, skipping...")
                        continue

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
                        if True:
                            batch['source_shading'] = batch['source_shading'] / args.max_shading
                            for i in range(len(batch['target_shading'])):
                                batch['target_shading'][i] = batch['target_shading'][i] / args.max_shading
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
                            # generate SD output directly
                            print("Generating SD output")
                            ret = pipe.relight(
                                source_image = None,
                                target_shading = batch['target_shading'][target_idx].to(device).to(MASTER_TYPE),
                                prompt=batch['text'][0],
                                source_shading = None,
                                latents = None,
                                num_inference_steps=50,
                            )
                            sd_output = ret['image']

                            # save image
                            image = torch.clamp(image, 0.0, 1.0).cpu()
                            filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                            torchvision.utils.save_image(image, os.path.join(crop_dir, f"{filename}.png"))
                            os.chmod(os.path.join(crop_dir, f"{filename}.png"), 0o777)
                            
                            sd_output = torch.clamp(sd_output, 0.0, 1.0).cpu()
                            filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
                            torchvision.utils.save_image(sd_output, os.path.join(sdoutput_dir, f"{filename}.jpg"))
                            os.chmod(os.path.join(sdoutput_dir, f"{filename}.jpg"), 0o777)
                            
                            control_image = batch['target_shading'][target_idx]
                            control_image = (control_image - control_image.min()) / (control_image.max() - control_image.min())
                            filename = f"{batch['name'][0].replace('/','-')}_{batch['word_name'][target_idx][0].replace('/','-')}"
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

                            tb_image = [gt_image, image, sd_output, control_image]
                            tb_image = torch.cat(tb_image, dim=0)
                            tb_image = torch.clamp(tb_image, 0.0, 1.0)
                            tb_image = torchvision.utils.make_grid(tb_image, nrow=int(np.ceil(np.sqrt(len(tb_image)))), normalize=True)
                            torchvision.utils.save_image(tb_image, os.path.join(with_groudtruth_dir, f"{filename}.jpg"))
                            os.chmod(os.path.join(with_groudtruth_dir, f"{filename}.jpg"), 0o777)


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
    parser.add_argument("-p", "--preset", type=str, default="1e-4")
    #parser.add_argument("-m", "--mode", type=str, default="rotate_everett_kitchen6,rotate_everett_dining1,rotate_everett_kitchen2,rotate_everett_kitchen4,everett_kitchen6,everett_kitchen4,everett_kitchen2,everett_dining1")
    #parser.add_argument("-m", "--mode", type=str, default="rotate_everett_kitchen6_diffusionlight_shading,rotate_everett_dining1_diffusionlight_shading,rotate_everett_kitchen2_diffusionlight_shading,rotate_everett_kitchen4_diffusionlight_shading,everett_dining1_diffusionlight_shading,everett_kitchen2_diffusionlight_shading,everett_kitchen4_diffusionlight_shading,everett_kitchen6_diffusionlight_shading")
    parser.add_argument("-m", "--mode", type=str, default="rotate_everett_kitchen6_least_square_shading")
    parser.add_argument("--max_shading",  type=float, default=32.0)
    parser.add_argument('-seed', type=str, default='42')
    args = parser.parse_args()
    main(args)

