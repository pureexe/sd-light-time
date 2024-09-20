#version 0, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 1, with depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 2, with bae
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 3, with bothbae
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 8

###########################################

#version 4, no_control (rerun)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 5, with depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 6, with bae
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 8

#version 7, with bothbae
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 8

############################

#version 8, with depth #continue from version 5
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000004.ckpt

#version 9, with bae #continue from version 6
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_6/checkpoints/epoch=000004.ckpt

#verison 10, with bothbae #continue from version 7
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 12 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_7/checkpoints/epoch=000019.ckpt

#verison 11, no_control #continue from version 4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 4 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000024.ckpt