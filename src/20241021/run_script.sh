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


# v6 depth - continue from v0
NODE=v9 GPU=0 NAME=depth bin/silocal src/20241021/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000234.ckpt

# v7 both_bae - continue from v1
NODE=v9 GPU=1 NAME=both_bae bin/silocal src/20241021/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch=000214.ckpt

# v8  bae from v4
NODE=v23 GPU=2 NAME=bae bin/silocal src/20241021/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000234.ckpt

# v8  no_control from v5
NODE=v23 GPU=3 NAME=no_control bin/silocal src/20241021/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 3.0 --batch_size 12 -ckpt output/20241021/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000259.ckpt

