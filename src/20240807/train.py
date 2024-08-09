import torch
from EnvmapAffine import EnvmapAffine
from UnsplashLiteDataset import UnsplashLiteDataset
import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR
from LineNotify import notify

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-em', '--envmap_embedder', type=str, default="vae") 
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('-b', '--train_begin', type=int, default=0)
parser.add_argument('-e', '--train_end', type=int, default=20000) 
parser.add_argument('-c', '--every_n_epochs', type=int, default=1) 
args = parser.parse_args()

@notify
def main():
    model = EnvmapAffine(learning_rate=args.learning_rate,envmap_embedder=args.envmap_embedder, face100_every=5)
    train_dataset = UnsplashLiteDataset(split=slice(args.train_begin, args.train_end, 1), root_dir=DATASET_ROOT_DIR+"/train")
    val_dataset = UnsplashLiteDataset(split=slice(0,20000,10000), root_dir=DATASET_ROOT_DIR+"/train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        filename="{epoch:06d}",
        every_n_epochs=args.every_n_epochs,
        save_top_k=-1,  # <--- this is important!
    )

    trainer = L.Trainer(
        max_epochs=10000,
        precision="16-mixed",
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        default_root_dir=OUTPUT_MULTI,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.checkpoint
    )

if __name__ == '__main__':
    main()