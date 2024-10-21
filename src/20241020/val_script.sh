NODE=v17 GPU=1 NAME=val_chk39_1 bin/silocal src/20241020/val_ddim.py -g 1.0,3.0,5.0,7.0


CUDA_VISIBLE_DEVICES=0 bin/py src/20241020/val_ddim.py -g 1.0,3.0,5.0,7.0