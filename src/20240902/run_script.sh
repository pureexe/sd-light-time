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
CUDA_VISIBLE_DEVICES=3 bin/py src/20240902/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_12/checkpoints/epoch=000015.ckpt

# version 15 - both_bae  1e-4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240902/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 10 -ckpt output/20240902/multi_mlp_fit/lightning_logs/version_13/checkpoints/epoch=000014.ckpt
