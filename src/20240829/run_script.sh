#version 0 - lr 5e-4 - LOSS NAN
bin/py src/20240829/train.py -lr 5e-4

#version 1 -lr 1e-4
bin/py src/20240829/train.py-lr 1e-4

#version 2 -lr 5e-5
bin/py src/20240829/train.py -lr 5e-5

#version 3 -lr 1e-5
bin/py src/20240829/train.py -lr 1e-5

#version 4 -lr 1e-4 / mult10
bin/py src/20240829/train.py -lr 1e-4 -gm 10

#version 5 -lr 1e-4 / mult10
bin/py src/20240829/train.py -lr 1e-4 -gm 100

#version 8 -lr 1e-4 -ct depth
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct both -gm 10

#version 9 affine normal
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct normal -gm 10


#version 10 affine normal depth
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct both

#version 11
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 5e-5 -ct both

#version 12
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-5 -ct both

#version 13 
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-5 -ct both --batch_size 10

#version 14
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 5e-5 -ct both --batch_size 10


#version 15
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 5e-5 -ct normal 
#version 16
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-5 -ct normal 


#version 17
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit1

#version 18
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit100

#version 19
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1

#version 20
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit100

#version 21
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit1

#version 22
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit100

#version 23
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1

#version 24
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit100

#version 25 loss without controlnet
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit1 --guidance_scale 3.0

#version 26
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct depth -split overfit100 --guidance_scale 3.0

#version 27
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1 --guidance_scale 3.0

#version 28
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit100 --guidance_scale 3.0

#version 29 checkval every 100
 CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1 --guidance_scale 3.0 --val_check_interval 1

#version 32 checkval every 100
 CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1 --guidance_scale 3.0 --val_check_interval 1

#version 33 checkval every 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct both -split overfit1 --guidance_scale 3.0 --val_check_interval 2

#version 34 checkval every 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct no_control -split overfit1 --guidance_scale 3.0 --val_check_interval 2

#version 35 checkval every 2 #AffineNoControl
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-4 -ct no_control -split overfit1 --guidance_scale 3.0 --val_check_interval 2

#version 36 checkval every 2 #AffineNoControl that use  old validation code
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-4 -ct no_control -split overfit1 --guidance_scale 3.0 --val_check_interval 2


#version 39 checkval every 2 #AffineNoControl that use  new validation code but old training code
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-4 -ct no_control -split overfit1 --guidance_scale 3.0 --val_check_interval 2

#version 43 old init instead (This is working)

#version 45 use new light block function

###################################
#version 48, without control net

CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct no_control --guidance_scale 3.0

# version 49 - depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct depth --guidance_scale 3.0

# version 50 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-4 -ct normal --guidance_scale 3.0

# version 51 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-4 -ct both --guidance_scale 3.0 --batch_size 10

#version 52, without control net

CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-5 -ct no_control --guidance_scale 3.0

# version 53 - depth
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-5 -ct depth --guidance_scale 3.0

# version 54 - normal
CUDA_VISIBLE_DEVICES=2 bin/py src/20240829/train.py -lr 1e-5 -ct normal --guidance_scale 3.0

# version 55 - both
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-5 -ct both --guidance_scale 3.0 --batch_size 10

# version 56 - normal bae
CUDA_VISIBLE_DEVICES=0 bin/py src/20240829/train.py -lr 1e-4 -ct bae --guidance_scale 3.0 --batch_size 10

# version 57 - both_bae # FAILED DUE TO DEPTH IS MISSING
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 10

# version 58 - bae1-e5
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-5 -ct bae --guidance_scale 3.0 --batch_size 10

# version 59 - bothbae1-e5 #FAILED DUE TO MISSING BAE
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 10

# version 60  both_bae1e-4
CUDA_VISIBLE_DEVICES=1 bin/py src/20240829/train.py -lr 1e-4 -ct both_bae --guidance_scale 3.0  --batch_size 10

# version 61  both_bae1e-5
CUDA_VISIBLE_DEVICES=3 bin/py src/20240829/train.py -lr 1e-5 -ct both_bae --guidance_scale 3.0 --batch_size 10