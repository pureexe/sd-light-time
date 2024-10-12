###################################
#version 0, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20240914/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0  --batch_size 16

# version 1 - depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240914/train.py -lr 1e-4 -ct depth --guidance_scale 3.0  --batch_size 16

# version 2 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240914/train.py -lr 1e-4 -ct normal --guidance_scale 3.0  --batch_size 16

# version 3 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240914/train.py -lr 1e-4 -ct both --guidance_scale 3.0  --batch_size 16

#version 4, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20240914/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0  --batch_size 8

# version 5- depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240914/train.py -lr 1e-5 -ct depth --guidance_scale 3.0  --batch_size 8

# version 6 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240914/train.py -lr 1e-5 -ct normal --guidance_scale 3.0  --batch_size 8

# version 7 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240914/train.py -lr 1e-5 -ct both --guidance_scale 3.0  --batch_size 8

# version 8 - normal bae
CUDA_VISIBLE_DEVICES=0 bin/py src/20240914/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 8

# version 9 - both_bae  
CUDA_VISIBLE_DEVICES=1 bin/py src/20240914/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 8

# version 10 - bae1-e5
CUDA_VISIBLE_DEVICES=2 bin/py src/20240914/train.py -lr 1e-5 -ct bae --guidance_scale 3.0 --batch_size 8

# version 11 - bothbae1-e5 #NOPE, it does has BAE
CUDA_VISIBLE_DEVICES=3 bin/py src/20240914/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 8

# version 12 - no_control continue from version 0
CUDA_VISIBLE_DEVICES=0 bin/py src/20240914/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0  --batch_size 16 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000254.ckpt

# version 13 - depth continue from version 1
CUDA_VISIBLE_DEVICES=1 bin/py src/20240914/train.py -lr 1e-4 -ct depth --guidance_scale 3.0  --batch_size 16 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_1/checkpoints/epoch=000224.ckpt

#version 14 - normal continue from version 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20240914/train.py -lr 1e-4 -ct normal --guidance_scale 3.0  --batch_size 16 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_2/checkpoints/epoch=000234.ckpt

#version 15 - both continue from version 3
CUDA_VISIBLE_DEVICES=3 bin/py src/20240914/train.py -lr 1e-4 -ct both --guidance_scale 3.0  --batch_size 16 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_3/checkpoints/epoch=000214.ckpt

#version 16, without control net
CUDA_VISIBLE_DEVICES=0 bin/py src/20240914/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0  --batch_size 8 -ckpt output/20240914/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000144.ckpt