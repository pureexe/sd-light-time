# version generate

#0
NODE=v9 GPU=0 NAME=depth bin/siat src/20241015/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 12
#1
NODE=v9 GPU=1 NAME=both_bae bin/siat src/20241015/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 3.0 --batch_size 12
# 4
NODE=v23 GPU=2 NAME=bae bin/siat src/20241015/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12
# 5
NODE=v23 GPU=3 NAME=no_control bin/siat src/20241015/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 3.0 --batch_size 12

# 6 (from v5 -> v3)
NODE=v23 GPU=2 NAME=bae bin/siat src/20241015/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12
