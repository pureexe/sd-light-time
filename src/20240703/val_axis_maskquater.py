from EnvmapAffine import EnvmapAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify

VERSION = 20
#VERSION = 20
#CHK_PTS = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199]
#CHK_PTS = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199, 219, 239, 259, 279, 299]
#CHK_PTS = [199, 219, 239, 259, 279, 299]
#CHK_PTS = [19, 39, 59, 79, 99, 119, 139, 159, 179, 199, 219, 239, 259, 279, 299]
#CHK_PTS = [179]
CHK_PTS = [179]
LRS = ['5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '1e-4', '1e-5', '5e-6', '1e-6', '5e-7', '1e-7', '1e-3', '1e-4', '7e-4', '3e-4', '1e-4', '3e-4', '5e-5']
NAMES = ['dinov2', 'dinov2', 'dinov2', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'vae', 'vae', 'vae', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet', 'slimnet', 'vae5000', 'vae5000', 'vae5000']
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']

ENV_MODE = "mask_left_right"

line = LineNotify()
line.send(f"val_axis.py started at version {VERSION}", with_hostname=True)

for chk_pt in CHK_PTS:
    CKPT_PATH = f"output/20240703/multi_mlp_fit/lightning_logs/version_{VERSION}/checkpoints/epoch={chk_pt:06d}.ckpt"
    model = EnvmapAffine.load_from_checkpoint(CKPT_PATH)
    model.set_guidance_scale(3.0)
    model.eval() # disable randomness, dropout, etc...

    if ENV_MODE == "mask_left_right":
        print("mask_quater")
        prompt_path = "datasets/face/face2000_single/prompts.json"
        from EnvmapAffineTestDataset import EnvmapAffineTestDataset
        for axis_name in ['mask_topright']:
            trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_axis_mask_left_right/{NAMES[VERSION]}/{LRS[VERSION]}/chk{chk_pt}/")
            #val_dataset = EnvmapAffineTestDataset(root_dir=f"datasets/validation/mask_envmap/{axis_name}", dataset_multiplier=10, prompt_path=prompt_path)
            val_dataset = EnvmapAffineTestDataset(root_dir=f"datasets/validation/small_stuido_light2/", dataset_multiplier=10, prompt_path=prompt_path)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

    else:
        for val_file in VAL_FILES:
            trainer = L.Trainer(max_epochs=1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_axis/{NAMES[VERSION]}/{LRS[VERSION]}/chk{chk_pt}/{val_file}")
            val_dataset = EnvmapAffineDataset(split="0:10", specific_file=""+val_file+".json", dataset_multiplier=10, val_hold=0)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
            trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

line.send(f"val_axis.py at DONE version {VERSION}", with_hostname=True)
