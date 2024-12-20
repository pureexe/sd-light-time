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

#############

# version 12 is from 20240914/version0 (no_control)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0  --batch_size 16 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000254.ckpt

# version 13 is from 20240914/version1 (depth)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --guidance_scale 3.0  --batch_size 16 --feature_type shcoeff_order2  -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch=000224.ckpt

# version 14 is from 20240914/version2 (normal)
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-4 -ct normal --guidance_scale 3.0  --batch_size 16 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000234.ckpt

# version 15 is from 20240914/version3 (both)
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-4 -ct both --guidance_scale 3.0  --batch_size 16 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_3/checkpoints/epoch=000214.ckpt

# version 16 is from 20240914/version4 (no_control)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0  --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000144.ckpt

# version 17 is from 20240914/version5 (depth)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-5 -ct depth --guidance_scale 3.0  --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000134.ckpt

# version 18 is from 20240914/version6 (normal)
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-5 -ct normal --guidance_scale 3.0  --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_6/checkpoints/epoch=000144.ckpt

# version 19 is from 20240914/version7 (both)
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-5 -ct both --guidance_scale 3.0  --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_7/checkpoints/epoch=000124.ckpt

# version 20 is from 20240914/version8 (normal_bae)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_8/checkpoints/epoch=000139.ckpt

# version 21 is from 20240914/version9 (both_bae)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_9/checkpoints/epoch=000129.ckpt

# version 22 is from 20240914/version10 (bae1-e5)
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-5 -ct bae --guidance_scale 3.0 --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_10/checkpoints/epoch=000144.ckpt

# version 23 is from 20240914/version11 (both_bae1-e5)
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_11/checkpoints/epoch=000244.ckpt

# version 24 is from version11 (no_control)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 4 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_11/checkpoints/epoch=000034.ckpt

# version 25 is from version8 (depth)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_8/checkpoints/epoch=000019.ckpt

# version 26 is from version10 (bothbae)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 12 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_10/checkpoints/epoch=000049.ckpt

# version 27 is from version9 (normal)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_9/checkpoints/epoch=000034.ckpt

# version 28 is from 20240914/version11 (both_bae1-e5)
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 8 --feature_type shcoeff_order2 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_11/checkpoints/epoch=000124.ckpt

##################

# version 29 is from version24 (no_control)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 4 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_24/checkpoints/epoch=000089.ckpt

# version 30 is from version25 (depth)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_25/checkpoints/epoch=000139.ckpt

# version 31 is from version26 (bothbae)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 12 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_26/checkpoints/epoch=000164.ckpt

# version 32 is from version27 (normal)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_27/checkpoints/epoch=000104.ckpt

##################################

# version 33 is from version 29
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/train.py -lr 1e-4 -ct no_control --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_29/checkpoints/epoch=000129.ckpt

# version 34 is from version 30 (NO CHECKPOIONT HERE)
#CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_30/checkpoints/epoch=000189.ckpt

# version 35 is from version 31
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/train.py -lr 1e-4 -ct both_bae --feature_type vae --guidance_scale 7.0 --batch_size 12 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_31/checkpoints/epoch=000244.ckpt

# version 36 is from version 32
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/train.py -lr 1e-4 -ct bae --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_32/checkpoints/epoch=000189.ckpt

# version 37 is from version 30
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 7.0 --batch_size 16 -ckpt output/20240918/multi_mlp_fit/lightning_logs/version_30/checkpoints/epoch=000189.ckpt