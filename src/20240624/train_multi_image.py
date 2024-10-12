import torch
from FaceSingleAxisAffine import FaceSingleAxisAffine
from FaceSingleAxisDataset import FaceSingleAxisDataset
import lightning as L

import argparse 

from constants import OUTPUT_MULTI

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
args = parser.parse_args()

def main():
    model = FaceSingleAxisAffine(
        learning_rate=args.learning_rate,
        viz_video=False,
        viz_image=True
    )
    train_dataset = FaceSingleAxisDataset(split="train", left_right_only=True)
    val_dataset = FaceSingleAxisDataset(split="val", left_right_only=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="{epoch:06d}",
        every_n_epochs=1,
        save_top_k=-1,  # <--- this is important!
    )

    trainer = L.Trainer(
        max_epochs=1000,
        precision=16,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        default_root_dir=OUTPUT_MULTI,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == '__main__':
    main()