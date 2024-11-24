####
# TOTEST: bin/sitatv100 test_dataset.py


import torch
from torch.utils.data import DataLoader
from datasets.DiffusionFaceRelightDataset import DiffusionFaceRelightDataset

DATASET_TRAIN_PATH ="/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/train"
DATASET_VIS_PATH ="/ist/ist-share/vision/relight/datasets/face/ffhq_defareli/train"

def main():


    dataset = DiffusionFaceRelightDataset(root_dir=DATASET_TRAIN_PATH)
    # test if it can get 60k image 
    assert len(dataset) == 60000, "training set size is 60k"

    # test load with dataloader with batch 4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in dataloader:
        # check for essential key in batch 
        for k in ['name', 'source_image', 'background', 'shading', 'diffusion_face', 'text']:
            assert k in batch
        # test if it return shape 616 
        break # we only test for 1 batch 




    # test if filename informat of "00000/00000"
    print("Test PASS!")

if __name__ == "__main__":
    main()