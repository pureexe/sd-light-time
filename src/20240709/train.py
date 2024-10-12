import torch
from AlbedoAffine import AlbedoAffine
from AlbedoDataset import AlbedoDataset
import lightning as L

import argparse 

from constants import OUTPUT_MULTI

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-em', '--envmap_embedder', type=str, default="dino2") 
args = parser.parse_args()

def main():
    model = AlbedoAffine(learning_rate=args.learning_rate,envmap_embedder=args.envmap_embedder)
    train_dataset = AlbedoDataset(split=slice(0, 2000, 1))
    val_dataset = AlbedoDataset(split=slice(0, 2000, 1000))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="{epoch:06d}",
        every_n_epochs=20,
        save_top_k=-1,  # <--- this is important!
    )

    trainer = L.Trainer(
        max_epochs=1000,
        precision=16,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        default_root_dir=OUTPUT_MULTI,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        #ckpt_path=OUTPUT_MULTI + "/lightning_logs/version_3/checkpoints/epoch=000062.ckpt"
    )

if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # Catch any exception (more specific is better)
        try:
            from LineNotify import LineNotify
            line = LineNotify()
            line.send(f"STOP 20240709 lr:{args.learning_rate} / em:{args.envmap_embedder}", with_hostname=True)
        except:
            pass
        raise  # Re-raise the original exception


    # 5e-5, 1e-5, 1e-4