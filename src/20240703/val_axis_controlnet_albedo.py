from EnvMapControlAlbedoAffine import EnvMapControlAlbedoAffine
from EnvmapAffineDataset import EnvmapAffineDataset
import lightning as L
import torch
import argparse 
from LineNotify import notify

@notify
def main():
    CONTROLNET_LR = "1e-4"
    CKPT_PATH = "output/20240703/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000159.ckpt"
    model = EnvMapControlAlbedoAffine.load_from_checkpoint(CKPT_PATH)
    model.load_controlnet(f"../controlnet-albedo/output/albedo_controlnet/face2000/lr{CONTROLNET_LR}/checkpoint-50000/controlnet")
    model.eval() # disable randomness, dropout, etc...

    val_dataset = EnvmapAffineDataset(split="0:1")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    
    for guidance_scale in [3]:
        trainer = L.Trainer(max_epochs =1000, precision=16, check_val_every_n_epoch=1, default_root_dir=f"output/20240703/val_axis_control_albedo_v2/vae/5e-5/chk159/g{guidance_scale:.2f}/albedo{CONTROLNET_LR}")
        print(f"guidance_scale: {guidance_scale}")
        model.set_guidance_scale(guidance_scale)
        # test (pass in the loader)
        trainer.test(model, dataloaders=val_dataloader, ckpt_path=CKPT_PATH)



if __name__ == "__main__":
    main()
    # from LineNotify import LineNotify
    # line = LineNotify()
    # try:
    #     main()
    #     line.send("val_axis_controlnet.py finished successfully", with_hostname=True)
    # except Exception as e:
    #     line.send("val_axis_controlnet.py FAILED", with_hostname=True)
    #     raise