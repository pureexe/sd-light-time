# version generate
#0
NODE=v23 GPU=1 NAME=deepfloyd bin/siat src/20241020/train.py -lr 1e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 12

NODE=v28 GPU=0 NAME=deepfloyd_5e-4 bin/siat4080 src/20241020/train.py -lr 5e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 1


NODE=v18 GPU=0 NAME=deepfloyd_5e-5 bin/silocal src/20241020/train.py -lr 5e-5 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8

NODE=v18 GPU=1 NAME=deepfloyd_5e-4 bin/silocal src/20241020/train.py -lr 5e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8
