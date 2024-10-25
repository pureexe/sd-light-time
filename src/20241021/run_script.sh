#0
NODE=v9 GPU=0 NAME=depth bin/silocal src/20241021/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 12
#1
NODE=v9 GPU=1 NAME=both_bae bin/silocal src/20241021/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 3.0 --batch_size 12
#2
NODE=v23 GPU=2 NAME=bae bin/silocal src/20241021/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12
#3
NODE=v23 GPU=3 NAME=no_control bin/silocal src/20241021/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 3.0 --batch_size 12

# v4 is resume from v2 

NODE=v23 GPU=2 NAME=bae bin/silocal src/20241021/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000189.ckpt

# v5 is resume from v3
NODE=v23 GPU=3 NAME=no_control bin/silocal src/20241021/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_3/checkpoints/epoch=000209.ckpt

# v23 - both_bae
# v23 - bae 
# v23 - deepfloyd
