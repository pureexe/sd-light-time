import torch
from FaceLeftRightAffine import FaceLeftRightAffine
from FaceLeftRightDataset import FaceLeftRightDataset
import lightning as L

import argparse 

from constants import OUTPUT_SINGLE

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
args = parser.parse_args()

def main():
    model = FaceLeftRightAffine(learning_rate=args.learning_rate)
    train_dataset = FaceLeftRightDataset(split="74:75", dataset_multiplier=100)
    val_dataset = FaceLeftRightDataset(split="74:75", dataset_multiplier=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    trainer = L.Trainer(max_epochs =1000, precision=32, check_val_every_n_epoch=1, default_root_dir=OUTPUT_SINGLE)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()