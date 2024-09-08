###################################
#version 0, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0

# version 1 - depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-4 -ct depth --guidance_scale 3.0

# version 2 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-4 -ct normal --guidance_scale 3.0

# version 3 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct both --guidance_scale 3.0 --batch_size 10

#version 4, without control net

CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0

# version 5 - depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-5 -ct depth --guidance_scale 3.0

# version 6 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-5 -ct normal --guidance_scale 3.0

# version 7 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-5 -ct both --guidance_scale 3.0 --batch_size 10

# version 8 - normal bae #CANCLE
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 10

# version 9 - both_bae  #CANCLE
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 10

# version 10 - bae1-e5
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-5 -ct bae --guidance_scale 3.0 --batch_size 10

# version 11 - bothbae1-e5 #FAILED DUE TO MISSING BAE
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 10

# version 12 - normal bae 1e-4
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 4

# version 13 - both_bae  1e-4
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 4


# version 14 - resume v12
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 4 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_12/checkpoints/epoch=000015.ckpt

# version 15 - both_bae  1e-4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 4 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_13/checkpoints/epoch=000014.ckpt

#version 16: no_control - continue from version 0
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000105.ckpt

# version 17: depth - continue from version 1
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-4 -ct depth --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch=000094.ckpt

# version 18: normal  - continue from version 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-4 -ct normal --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000098.ckpt

# version 19: both  - continue from version 3
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct both --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_3/checkpoints/epoch=000085.ckpt

#version 20: no_controlnet - continue from 4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000106.ckpt

# version 21: depth - continue from 5 
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-5 -ct depth --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000092.ckpt

# version 22: normal continue from 6
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 1e-5 -ct normal --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_6/checkpoints/epoch=000096.ckpt

# version 23: both continue from 7
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-5 -ct both --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_7/checkpoints/epoch=000083.ckpt

# version 24: normal bae 1e-4 from 12
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_12/checkpoints/epoch=000032.ckpt

# version 25: both bae 1e-4 from 13
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_13/checkpoints/epoch=000030.ckpt

#version 26: no_control - continue from 10
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-5 -ct bae --guidance_scale 3.0 --batch_size 8 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_10/checkpoints/epoch=000154.ckpt

#version 27: no_control - continue from 11
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 8 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_11/checkpoints/epoch=000143.ckpt

#version 28 face5e-4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 5e-4 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face

#version 29 face1e-4
CUDA_VISIBLE_DEVICES=1 bin/py src/20240902/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face

#version 30 face5e-5
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 5e-5 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face

#version 31 face1e-5
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face

#version 32 face1e-3
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-3 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face

#version 33 face5e-3
CUDA_VISIBLE_DEVICES=2 bin/py src/20240902/train.py -lr 5e-3 -ct no_control --guidance_scale 3.0 --batch_size 4 --every_n_epochs 20 --dataset "/data/pakkapon/datasets/face/face2000" -split train_face