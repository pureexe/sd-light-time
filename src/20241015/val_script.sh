NODE=v23 GPU=0 NAME=val_depth bin/siat src/20241015/val_ddim_mix.py

NODE=v23 GPU=1 NAME=val_depth_checkpoint bin/siat src/20241015/val_ddim_mix.py -c 4,9,14,19,24,29,34,39,44,49,54,59,64

NODE=v23 GPU=0 NAME=val_depth_checkpoint bin/siat src/20241015/val_ddim_mix.py -c 0