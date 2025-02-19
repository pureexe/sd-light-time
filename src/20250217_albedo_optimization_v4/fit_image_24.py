from albedo_optimization import MultiIluminationSceneDataset, AlbedoOptimization
import argparse

import torch
import lightning as L

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
    parser.add_argument('-std_mul', '--std_multiplier', type=float, default=1e-4)
    parser.add_argument('-lra', '--lr_albedo', type=float, default=1e-3)
    parser.add_argument('-lrs', '--lr_shcoeff', type=float, default=1e-3)
    parser.add_argument('--dataset_multipiler', type=int, default=100)
    parser.add_argument('--sh_regularize', type=float, default=1e-3)
    parser.add_argument('--sh_3channel', type=float, default=0)
    parser.add_argument('--cold_start_albedo', type=int, default=1000000, help="epoch to start training albedo, 0 mean start training since first epoch")
    parser.add_argument('--use_lab', type=int, default=1)
    args = parser.parse_args()

def main():
    SCENE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/images/14n_copyroom10"
    full_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=1,
        image_size=(512,512),
        use_lab = args.use_lab == 1
    )
    train_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=args.dataset_multipiler,
        image_size=(512,512),
        use_lab = args.use_lab == 1
    )
    val_dataset = MultiIluminationSceneDataset(
        scene_path=SCENE_DIR,
        data_multiplier=1,
        image_size=(512,512),
        use_lab = args.use_lab == 1
    )
    train_dataset.image_paths = train_dataset.image_paths[24:25]
    val_dataset.image_paths = val_dataset.image_paths[24:25]

    model = AlbedoOptimization(
        num_images = train_dataset.get_num_images(),
        lr_albedo = args.lr_albedo,
        lr_shcoeff = args.lr_shcoeff,
        std_multiplier = args.std_multiplier,
        cold_start_albedo = args.cold_start_albedo,
        sh_regularize = args.sh_regularize,
        sh_3channel = args.sh_3channel,
        use_lab = args.use_lab == 1
    )
    model.initial_with_mean(full_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.get_num_images(), shuffle=False, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.get_num_images(), shuffle=False)
    trainer = L.Trainer(reload_dataloaders_every_n_epochs=0)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.checkpoint
    )

if __name__ == "__main__":
    main()