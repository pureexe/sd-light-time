import torch

from datasets.DDIMArrayEnvDataset import DDIMArrayEnvDataset
from datasets.RelightDataset import RelightDataset

import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sdrelightwithchromeball import SDRelightWithChromeball

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('-c', '--every_n_epochs', type=int, default=1) 
parser.add_argument('--feature_type', type=str, default='shcoeff_order2')
parser.add_argument('-gm', '--gate_multipiler', type=float, default=1)
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument('--dataset_train_multiplier', type=int, default=1)
parser.add_argument('-guidance', '--guidance_scale', type=float, default=1.0) 
parser.add_argument('-dataset', '--dataset', type=str, default=DATASET_ROOT_DIR) 
parser.add_argument('-dataset_val', '--dataset_val', type=str, default=DATASET_VAL_DIR) 
parser.add_argument('-dataset_val_split', '--dataset_val_split', type=str, default=DATASET_VAL_SPLIT) 
parser.add_argument('-specific_prompt', type=str, default="")  # we use static prompt to make thing same as mint setting

parser.add_argument(
    '-split',  
    type=str,
    choices=['none','overfit1', 'overfit100','train_face'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    default='none'
)
args = parser.parse_args()

def get_model_class():
    return SDRelightWithChromeball


def main():
    model_class = get_model_class()
    model = model_class(
        learning_rate=args.learning_rate,
        guidance_scale=args.guidance_scale,
        feature_type=args.feature_type,
    )
    train_dir = args.dataset
    val_dir = args.dataset_val 
    specific_prompt = args.specific_prompt if args.specific_prompt != "" else None
    train_dataset = RelightDataset(root_dir=train_dir, dataset_multiplier=args.dataset_train_multiplier,specific_prompt=specific_prompt)
    val_dataset = DDIMArrayEnvDataset(root_dir=val_dir, index_file=args.dataset_val_split,specific_prompt=specific_prompt)
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
        num_sanity_val_steps=0
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