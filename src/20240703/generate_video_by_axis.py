# create image size 11x2 
import os 
import torch
import torchvision
import skimage
import numpy as np
import skimage.transform


DATASET_DIR = "datasets/validation/rotate_point/axis_x"
for scene in ['vae5000/1e-4', 'vae5000/3e-4', 'vae5000/5e-5']:
    for chk in [59, 79, 99]:
        for axis_name in ['axis_x', 'axis_y', 'axis_z']:
            print(scene, chk, axis_name)
            for image_id in range(24):
                #rows = []
                row = []
                directory = f"../../output/20240703/val_axis_lightdirection/{scene}/chk{chk}/{axis_name}" 
                os.makedirs(f"{directory}/lightning_logs/version_0/compare", exist_ok=True)
                # load ground truth image
                gt_image = skimage.io.imread(f"../../datasets/validation/rotate_point/{axis_name}/images/{image_id:04d}.png")
                gt_image = skimage.img_as_float(gt_image)
                if gt_image.shape[-1] == 4:
                    # blend with black background
                    gt_image = gt_image[..., :3] * gt_image[..., 3:]# + (1 - gt_image[..., 3:])
                gt_image = skimage.transform.resize(gt_image, (256, 256))
                #gt_image = mark_brightness_color(gt_image)
                gt_image = torch.tensor(gt_image).permute(2, 0, 1)
                row.append(gt_image)
                for scene_id in range(10):
                    #img = torchvision.io.read_image(f"{directory}/lightning_logs/version_0/crop_image/{image_id:04d}_{scene_id:05d}.png") / 255.0
                    img = skimage.io.imread(f"{directory}/lightning_logs/version_0/crop_image/{image_id:04d}_{scene_id:05d}.png")
                    img = skimage.img_as_float(img)
                    img = skimage.transform.resize(img, (256, 256))
                    #img = mark_brightness_color(img)
                    img = torch.tensor(img).permute(2, 0, 1)
                    row.append(img)
                
                # load ground truth ball
                gt_ball = skimage.io.imread(f"../../datasets/validation/rotate_point/{axis_name}/ball/{image_id:04d}.png")
                gt_ball = skimage.img_as_float(gt_ball)
                if gt_ball.shape[-1] == 4:
                    # blend with black background
                    gt_ball = gt_ball[..., :3] * gt_ball[..., 3:]# + (1 - gt_ball[..., 3:])
                gt_ball = skimage.transform.resize(gt_ball, (256, 256))
                gt_ball = mark_brightness_color(gt_ball)
                gt_ball = torch.tensor(gt_ball).permute(2, 0, 1)
                row.append(gt_ball)
                
                # load gt image with torchvision
                for scene_id in range(10):
                    #img = torchvision.io.read_image(f"{directory}/lightning_logs/version_0/ball/{image_id:04d}_{scene_id:05d}.png") / 255.0
                    img = skimage.io.imread(f"{directory}/lightning_logs/version_0/ball/{image_id:04d}_{scene_id:05d}.png")
                    img = skimage.img_as_float(img)
                    img = skimage.transform.resize(img, (256, 256))
                    img = mark_brightness_color(img)
                    img = torch.tensor(img).permute(2, 0, 1)
                    row.append(img)

                image = torchvision.utils.make_grid(row, nrow=11)
                torchvision.utils.save_image(image, f"{directory}/lightning_logs/version_0/compare/{image_id:04d}.png")
