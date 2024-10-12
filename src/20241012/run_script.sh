# here we will train network static 1 prompt prompt

#version 0, with depth at v9
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 8

#version 1, with bothbae at v9
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 3.0 --batch_size 8

#version 2, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 3.0 --batch_size 8



#version 3, with bae (CANCLE)
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 3.0 --batch_size 12





