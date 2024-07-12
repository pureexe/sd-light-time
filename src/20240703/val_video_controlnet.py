from EnvMapControlAffine import EnvMapControlAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch

def main():

    CKPT_PATH = "output/20240703/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000299.ckpt"
    model = EnvMapControlAffine.load_from_checkpoint(CKPT_PATH)
    model.eval() # disable randomness, dropout, etc...

    val_dataset = EnvmapAffineDataset(split="0:1")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    for guidance_scale in [5]:
        trainer = L.Trainer(max_epochs =1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_video_control/vae/5e-5/chk299/g{guidance_scale:.2f}")
        print(f"guidance_scale: {guidance_scale}")
        model.set_guidance_scale(guidance_scale)
        # test (pass in the loader)
        trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)

if __name__ == "__main__":
    from LineNotify import LineNotify
    line = LineNotify()
    try:
        main()
        line.send("val_video_controlnet.py finished successfully", with_hostname=True)
    except Exception as e:
        line.send("val_video_controlnet.py FAILED", with_hostname=True)
        raise