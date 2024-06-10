import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath('..'))

from FaceLeftRightDataset import FaceLeftRightDataset

def test_faceleftrightdataset():
    train_dataset = FaceLeftRightDataset(split="train")
    val_dataset = FaceLeftRightDataset(split="val")
    assert len(train_dataset) == 1900
    assert len(val_dataset) == 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

