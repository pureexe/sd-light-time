import torch

from AffineCondition import AffineDepth, AffineNormal, AffineNormalBae, AffineDepthNormal, AffineDepthNormalBae, AffineNoControl

from datasets.RelightDataset import RelightDataset
import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR

from LineNotify import notify

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('-b', '--train_begin', type=int, default=0)
parser.add_argument('-e', '--train_end', type=int, default=500000) 
parser.add_argument('-c', '--every_n_epochs', type=int, default=5) 
parser.add_argument('--feature_type', type=str, default='shcoeff_order2')
parser.add_argument('-gm', '--gate_multipiler', type=float, default=1)
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument(
    '-ct', 
    '--control_type', 
    type=str,
    choices=['depth','normal', 'both','no_control', 'bae', 'both_bae'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    required=True
)

parser.add_argument('-guidance', '--guidance_scale', type=float, default=1.0) 
parser.add_argument('-dataset', '--dataset', type=str, default=DATASET_ROOT_DIR) 
parser.add_argument('-dataset_val', '--dataset_val', type=str, default=DATASET_VAL_DIR) 
parser.add_argument(
    '-split', 
    type=str,
    choices=['none','overfit1', 'overfit100','train_face'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    default='none'
)
args = parser.parse_args()

def get_val_split():
    with open("src/20240829/id_left_right_20.txt") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    return content

def get_model_class():
    if args.control_type == 'depth':
        return AffineDepth
    elif args.control_type == 'normal':
        return AffineNormal
    elif args.control_type == 'both':
        return AffineDepthNormal
    elif args.control_type == 'both_bae':
        return AffineDepthNormalBae
    elif args.control_type == 'no_control':
        return AffineNoControl
    elif args.control_type == 'bae':
        return AffineNormalBae

@notify
def main():
    model_class = get_model_class()
    model = model_class(learning_rate=args.learning_rate, gate_multipiler=args.gate_multipiler, guidance_scale=args.guidance_scale)
    train_dir = args.dataset
    val_dir = args.dataset_val 
    train_dataset = RelightDataset(root_dir=train_dir, dataset_multiplier=1)
    val_dataset = RelightDataset(root_dir=val_dir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
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
        val_check_interval=args.val_check_interval
    )
    if not args.checkpoint or not os.path.exists(args.checkpoint):
       trainer.validate(model, val_dataloader)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.checkpoint
    )

    

if __name__ == '__main__':
    main()