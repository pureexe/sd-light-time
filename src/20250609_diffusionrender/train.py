import torch

from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset

import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sddiffusionrenderer import SDDiffusionRenderer

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-lr_expo_decay', '--lr_expo_decay', type=float, default=1.0)

parser.add_argument('--restart_ckpt', type=int, default=0, help="restart ckpt parameter back to epoch 0, we need to do this in case you want to resume model  with differnet leraning rate")
parser.add_argument('-clr', '--ctrlnet_lr', type=float, default=1)
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lora_rank', type=int, default=256)
parser.add_argument('-c', '--every_n_epochs', type=int, default=1) 
parser.add_argument('--feature_type', type=str, default='diffusion_face')
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument('--bg_mask_ratio', type=float, default=0.25)
parser.add_argument('--dataset_train_multiplier', type=int, default=1)
parser.add_argument('-guidance', '--guidance_scale', type=float, default=1.0) 
parser.add_argument('-dataset', '--dataset', type=str, default=DATASET_ROOT_DIR) 
parser.add_argument('-dataset_split', type=str, default="")
parser.add_argument('-dataset_val', '--dataset_val', type=str, default=DATASET_VAL_DIR) 
parser.add_argument('-dataset_val_split', '--dataset_val_split', type=str, default=DATASET_VAL_SPLIT) 
parser.add_argument('-specific_prompt', type=str, default="a photorealistic image")  # we use static prompt to make thing same as mint setting
parser.add_argument('--shadings_dir', type=str, default="shadings")
parser.add_argument('--backgrounds_dir', type=str, default="backgrounds") 
parser.add_argument('--images_dir', type=str, default="images") 
parser.add_argument('--false_shading', action='store_true', help='Set false_shading to True')
parser.add_argument('--triplet_background', action='store_true', help='Compute triplet loss for background to avoid cheating.')
parser.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation if need')
parser.add_argument('--seed', type=int, default=42, help='seed to use')


parser.add_argument(
    '-split',  
    type=str,
    choices=['none','overfit1', 'overfit100','train_face'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    default='none'
)
args = parser.parse_args()

def get_model_class():
    return SDDiffusionRenderer

def main():
    L.seed_everything(args.seed)
    model_class = get_model_class()
    if args.checkpoint is None:
        model = model_class(
            lr_light_encoder=args.learning_rate,
            lr_conv_in=args.learning_rate,
            lora_params=args.learning_rate,
            lora_rank=args.lora_rank,
        )
    else:
        # add learning rate parameter in case we load with differnet learning
        model = model_class.load_from_checkpoint(args.checkpoint, learning_rate=args.learning_rate)
    
                
    train_dir = args.dataset
    val_dir = args.dataset_val 

    specific_prompt = args.specific_prompt if args.specific_prompt not in ["","_"] else None

    train_dataset = DiffusionRendererEnvmapDataset(
        root_dir=train_dir,
    )
    val_dataset = DiffusionRendererEnvmapDataset(
        root_dir=val_dir,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        #filename="epoch{epoch:06d}_step{step:06d}",
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
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.grad_accum
    )
    # if not args.checkpoint or not os.path.exists(args.checkpoint):
    #    trainer.validate(model, val_dataloader)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path= args.checkpoint if args.restart_ckpt == 0 else None
    )

    

if __name__ == '__main__':
    main()