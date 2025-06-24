import torch

from vll_datasets.DiffusionRendererEnvmapDataset import DiffusionRendererEnvmapDataset
from vll_datasets.DiffusionRendererEnvmapDDIMDataset import DiffusionRendererEnvmapDDIMDataset

import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sdrelightenv import SDRelightEnv, SDAlbedoNormalDepthRelightEnv, SDAlbedoNormalDepthRelightIrradientEnv, SDRelightIrradientEnv, SDNormalDepthRelightEnv

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-mul_lr_gate', type=float, default=1.0, help="how much learning rate to multiply gate")
parser.add_argument('-lr_expo_decay', '--lr_expo_decay', type=float, default=1.0)

parser.add_argument('--restart_ckpt', type=int, default=0, help="restart ckpt parameter back to epoch 0, we need to do this in case you want to resume model  with differnet leraning rate")
parser.add_argument('-clr', '--ctrlnet_lr', type=float, default=1)
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=4)
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
parser.add_argument('--grad_accum', type=int, default=8, help='gradient accumulation if need')
parser.add_argument('--seed', type=int, default=42, help='seed to use')
parser.add_argument('--fade_step', type=int, default=50000, help='seed to use')
parser.add_argument('--lora_rank', type=int, default=256, help='rank of the lora')
parser.add_argument(
    '-nt', 
    '--network_type', 
    type=str,
    choices=['default', 'albedo_normal_depth', 'irradiant', 'albedo_normal_depth_irradiant', 'normal_depth'],
    help="select control type for the model",
    default='default'
)

parser.add_argument(
    '-split',  
    type=str,
    choices=['none','overfit1', 'overfit100','train_face'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    default='none'
)
args = parser.parse_args()

def get_model_class():
    if args.network_type == 'default':
        return SDRelightEnv
    elif args.network_type == 'albedo_normal_depth':
        return SDAlbedoNormalDepthRelightEnv
    elif args.network_type == 'albedo_normal_depth_irradiant':
        return SDAlbedoNormalDepthRelightIrradientEnv
    elif args.network_type == 'irradiant':
        return SDRelightIrradientEnv
    elif args.network_type == 'normal_depth':
        return  SDNormalDepthRelightEnv
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")

def get_dataset_components():
    if args.network_type == 'default':
        return ['envmap_feature', 'image']
    elif args.network_type == 'albedo_normal_depth':
        return ['albedo', 'normal', 'depth', 'envmap_feature', 'image']
    elif args.network_type == 'albedo_normal_depth_irradiant':
        raise ValueError("albedo_normal_depth_irradiant is not supported in this script, use other directory instead instead")
        return ['albedo', 'normal', 'depth', 'envmap_feature', 'image', 'irradiant']
    elif args.network_type == 'irradiant':
        raise ValueError("irradiant is not supported in this script, use other directory instead instead")
        return ['light', 'image', 'irradiant']
    elif args.network_type == 'normal_depth':
        return ['normal', 'depth', 'envmap_feature', 'image']
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")

def main():
    L.seed_everything(args.seed)
    model_class = get_model_class()
    if args.checkpoint is None:
        model = model_class(
            learning_rate=args.learning_rate,
            mul_lr_gate=args.mul_lr_gate,
            lora_rank=args.lora_rank,
            fade_step=args.fade_step
        )
    else:
        # add learning rate parameter in case we load with differnet learning
        model = model_class.load_from_checkpoint(args.checkpoint, learning_rate=args.learning_rate)
    
                
    train_dir = args.dataset
    val_dir = args.dataset_val 

    dataset_components = get_dataset_components()

    train_dataset = DiffusionRendererEnvmapDataset(
        root_dir=train_dir,
        components=dataset_components
    )
    val_dataset = DiffusionRendererEnvmapDDIMDataset(
        root_dir=val_dir,
        index_file=DATASET_VAL_SPLIT,
        components=dataset_components,
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