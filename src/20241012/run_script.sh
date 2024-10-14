# here we will train network static prompt


#version 2, with depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 3, with bothbae
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 8


#version 4, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 8


#version 5, with bae
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 8

