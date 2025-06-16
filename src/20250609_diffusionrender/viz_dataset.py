import torch
import os
from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset
from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
import skimage

OUTPUT_DIR = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val_viz"
KEYS = ['albedo', 'depth', 'normal', 'light_ldr', 'light_log_hdr', 'light_dir', 'image']

def main():
    train_dataset = DiffusionRendererEnvmapDataset(
        root_dir=DATASET_VAL_DIR,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    for batch in train_dataloader:
        name = batch['name'][0]
        for key in KEYS:
            os.makedirs("{}/{}".format(OUTPUT_DIR, key), exist_ok=True)
            output_path = "{}/{}/{}.png".format(OUTPUT_DIR, key, name)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            data = batch[key][0].cpu()
            data = data.permute(1, 2, 0)
            data = data.numpy()
            print(data.shape, key, name)
            data = (data + 1.0) / 2.0
            data = skimage.img_as_ubyte(data)
            skimage.io.imsave(output_path, data)
            
        print(f"Saved images for {name} in {OUTPUT_DIR}")
        


if __name__ == "__main__":
    main()