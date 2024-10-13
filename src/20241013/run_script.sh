# version generate
NODE=v28 GPU=0 NAME=vae256 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=1 NAME=vae128 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae128 --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=2 NAME=vae64 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae64 --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=3 NAME=vae32 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae32 --guidance_scale 3.0 --batch_size 8

