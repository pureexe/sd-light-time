import os 
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse
from termcolor import colored

from albedo_optimization import MultiIluminationSceneDataset, AlbedoOptimization

import warnings
warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('-std_mul', '--std_multiplier', type=float, default=1e-4)
parser.add_argument('-lra', '--lr_albedo', type=float, default=1e-2)
parser.add_argument('-lrs', '--lr_shcoeff', type=float, default=1e-2)
parser.add_argument('--dataset_multipiler', type=int, default=100)
parser.add_argument('--sh_regularize', type=float, default=1e-3)
parser.add_argument('--cold_start_albedo', type=int, default=0, help="epoch to start training albedo, 0 mean start training since first epoch")
parser.add_argument('--use_lab', type=int, default=0)
parser.add_argument('-i','--idx', type=int, default=0)
parser.add_argument('-t','--total', type=int, default=1)
args = parser.parse_args()


def main():
    SPLIT = "test"
    SCENE_DIR = f"/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/{SPLIT}/"
    scenes = sorted(os.listdir(SCENE_DIR+"/images"))
    scenes = scenes[args.idx::args.total]
    for scene_id, scene in enumerate(scenes):
        try:
            early_stopping = EarlyStopping(monitor="val/loss", patience=5, min_delta=1e-4, mode="min", verbose=True)
            default_root_dir = f"output/compute_albedo/{SPLIT}/{scene}"
            if os.path.exists(default_root_dir):
                continue
            print("=====================================================")
            print("SCENE: ", colored(scene,'green'), "  ( ",scene_id, " /  ", len(scenes), " )")
            print("=====================================================")
            image_dir = os.path.join(SCENE_DIR, "images", scene)
            train_dataset = MultiIluminationSceneDataset(
                scene_path=image_dir,
                data_multiplier=args.dataset_multipiler,
                image_size=(512,512),
                use_lab = args.use_lab == 1
            )
            val_dataset = MultiIluminationSceneDataset(
                scene_path=image_dir,
                data_multiplier=1,
                image_size=(512,512),
                use_lab = args.use_lab == 1
            )
            model = AlbedoOptimization(
                num_images = train_dataset.get_num_images(),
                lr_albedo = args.lr_albedo,
                lr_shcoeff = args.lr_shcoeff,
                std_multiplier = args.std_multiplier,
                cold_start_albedo = args.cold_start_albedo,
                sh_regularize = args.sh_regularize,
                use_lab = args.use_lab == 1,
                log_shading = False
            )
            model.initial_with_mean(val_dataset)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.get_num_images(), shuffle=False, num_workers=8)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.get_num_images(), shuffle=False)
            trainer = L.Trainer(
                reload_dataloaders_every_n_epochs=0,
                default_root_dir = default_root_dir,
                callbacks=[early_stopping],
                max_epochs=10000
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=args.checkpoint
            )
            model.save_shcoeffs()
            val_dataloader2 = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.get_num_images(), shuffle=False)
            for batch in val_dataloader2:
                model.save_shading(image=batch['image'][0])
                model.save_render(image=batch['image'][0])
                break
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    main()