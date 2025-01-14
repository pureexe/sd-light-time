import torch

from datasets.DDIMDiffusionFaceRelightDataset import DDIMDiffusionFaceRelightDataset
from datasets.DiffusionFaceRelightDataset import DiffusionFaceRelightDataset

import lightning as L

import argparse 
import os

from constants import OUTPUT_MULTI, DATASET_ROOT_DIR, DATASET_VAL_DIR, DATASET_VAL_SPLIT
from sddiffusionface import SDDiffusionFace, ScrathSDDiffusionFace, SDWithoutAdagnDiffusionFace, SDOnlyAdagnDiffusionFace, SDDiffusionFaceNoBg, SDDiffusionFaceNoShading, SDOnlyShading, SDDiffusionFace5ch

from LineNotify import notify

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-clr', '--ctrlnet_lr', type=float, default=1)
parser.add_argument('-ckpt', '--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-c', '--every_n_epochs', type=int, default=1) 
parser.add_argument('--feature_type', type=str, default='diffusion_face')
parser.add_argument('-gm', '--gate_multipiler', type=float, default=1)
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument('--bg_mask_ratio', type=float, default=0.25)
parser.add_argument('--dataset_train_multiplier', type=int, default=1)
parser.add_argument(
    '-nt', 
    '--network_type', 
    type=str,
    choices=['sd','scrath', 'sd_without_adagn', 'sd_only_adagn', 'sd_no_bg', 'sd_no_shading', 'sd_only_shading', 'inpaint', 'inpaint_no_shading', 'sd5ch', 'sd_color_jitter_defareli', 'sd_color_jitter'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    required=True
)
parser.add_argument('-guidance', '--guidance_scale', type=float, default=1.0) 
parser.add_argument('-dataset', '--dataset', type=str, default=DATASET_ROOT_DIR) 
parser.add_argument('-dataset_split', type=str, default="")
parser.add_argument('-dataset_val', '--dataset_val', type=str, default=DATASET_VAL_DIR) 
parser.add_argument('-dataset_val_split', '--dataset_val_split', type=str, default=DATASET_VAL_SPLIT) 
parser.add_argument('-specific_prompt', type=str, default="a photorealistic image")  # we use static prompt to make thing same as mint setting
parser.add_argument('--shadings_dir', type=str, default="shadings")
parser.add_argument('--backgrounds_dir', type=str, default="backgrounds") 

parser.add_argument(
    '-split',  
    type=str,
    choices=['none','overfit1', 'overfit100','train_face'],  # Restrict the input to the accepted strings
    help="select control type for the model",
    default='none'
)
args = parser.parse_args()

def get_model_class():
    if args.network_type in ['sd', 'inpaint', 'sd_color_jitter_defareli']:
        return SDDiffusionFace
    elif args.network_type == 'scrath':
        return ScrathSDDiffusionFace
    elif args.network_type in ['sd_without_adagn', 'sd_color_jitter']: # only controlnet part, both shading and background
        return SDWithoutAdagnDiffusionFace
    elif args.network_type == 'sd_only_adagn':
        return SDOnlyAdagnDiffusionFace
    elif args.network_type == 'sd_no_bg':
        return SDDiffusionFaceNoBg
    elif args.network_type in ['sd_no_shading','inpaint_no_shading']: # still has controlnet but wihtout shading
        return SDDiffusionFaceNoShading
    elif args.network_type == 'sd_only_shading': # only training the shading controlnet
        return SDOnlyShading
    elif args.network_type == 'sd5ch':
        return SDDiffusionFace5ch
@notify
def main():
    model_class = get_model_class()
    model = model_class(
        learning_rate=args.learning_rate,
        gate_multipiler=args.gate_multipiler,
        guidance_scale=args.guidance_scale,
        feature_type=args.feature_type,
        ctrlnet_lr=args.ctrlnet_lr
    )
    train_dir = args.dataset
    val_dir = args.dataset_val 
    use_shcoeff2 = args.feature_type in ['diffusion_face_shcoeff', 'clip_shcoeff', 'shcoeff_order2', 'vae_shcoeff']
    use_random_mask_background = args.network_type in ['inpaint_no_shading', 'inpaint'] 
    feature_types = ['shape', 'cam', 'faceemb', 'shadow']
    use_ab_background=args.network_type in ['sd5ch']
    if args.feature_type in ['shcoeff_order2']:
        feature_types = ['light']
    if args.feature_type in ['clip', 'vae', 'clip_shcoeff', 'vae_shcoeff']:
        print("WHY NOT HIT HERE?")
        feature_types = []
    if use_shcoeff2:
        feature_types.append('light')
    
    specific_prompt = args.specific_prompt if args.specific_prompt not in ["","_"] else None

    train_dataset = DiffusionFaceRelightDataset(
        root_dir=train_dir,
        index_file=args.dataset_split,
        dataset_multiplier=args.dataset_train_multiplier,
        specific_prompt=specific_prompt,
        use_shcoeff2=use_shcoeff2,
        feature_types=feature_types,
        random_mask_background_ratio=args.bg_mask_ratio if use_random_mask_background else None,
        shadings_dir=args.shadings_dir,
        backgrounds_dir=args.backgrounds_dir,
        use_ab_background=use_ab_background,
        use_background_jitter=args.network_type in ['sd_color_jitter', 'sd_color_jitter_defareli']
    )
    val_dataset = DDIMDiffusionFaceRelightDataset(
        root_dir=val_dir,
        index_file=args.dataset_val_split,
        specific_prompt=specific_prompt,
        use_shcoeff2=use_shcoeff2,
        feature_types=feature_types,
        random_mask_background_ratio = 0.0 if use_random_mask_background else None,
        shadings_dir=args.shadings_dir,
        backgrounds_dir=args.backgrounds_dir,
        use_ab_background=use_ab_background
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