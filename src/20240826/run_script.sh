#version 0
CUDA_VISIBLE_DEVICES=0 bin/py src/20240824/train.py -lr 5e-4  
#version 1 #########
CUDA_VISIBLE_DEVICES=1 bin/py src/20240824/train.py -lr 1e-4  
#version 2 #########
CUDA_VISIBLE_DEVICES=2 bin/py src/20240824/train.py -lr 5e-5   
#version 3
CUDA_VISIBLE_DEVICES=3 bin/py src/20240824/train.py -lr 1e-5

# version 5 - continue training version 1 
CUDA_VISIBLE_DEVICES=1 bin/py src/20240824/train.py -lr 1e-4  -ckpt output/20240824/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch=000029.ckpt

#version6 - continue training version 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20240824/train.py -lr 5e-5  -ckpt output/20240824/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000030.ckpt
