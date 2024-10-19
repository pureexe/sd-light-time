# version generate
#0
NODE=v23 GPU=1 NAME=deepfloyd bin/siat src/20241020/train.py -lr 1e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8
