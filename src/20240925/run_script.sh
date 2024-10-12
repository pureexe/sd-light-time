#version 0, face 1e-4
CUDA_VISIBLE_DEVICES=0 bin/py src/20240925/train.py -lr 1e-4 -ct depth --guidance_scale 7.0 --batch_size 1 -c 20 --feature_type shcoeff_order2 

#version 1, face 1e-5
CUDA_VISIBLE_DEVICES=1 bin/py src/20240925/train.py -lr 1e-5 -ct depth --guidance_scale 7.0 --batch_size 1 -c 20 --feature_type shcoeff_order2

#version 2, face 1e-4
CUDA_VISIBLE_DEVICES=2 bin/py src/20240925/train.py -lr 1e-4 -ct depth --guidance_scale 7.0 --batch_size 1 -c 20 --feature_type vae 

#version 3, face 1e-5
CUDA_VISIBLE_DEVICES=3 bin/py src/20240925/train.py -lr 1e-5 -ct depth --guidance_scale 7.0 --batch_size 1 -c 20 --feature_type vae
