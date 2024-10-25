NODE=v23 GPU=0 NAME=face1e-4 bin/silocal src/20241025/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16

NODE=v26 GPU=3 NAME=face1e-5 bin/siat4080 src/20241025/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4

NODE=v27 GPU=3 NAME=face5e-5 bin/siat4080 src/20241025/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4

NODE=v28 GPU=3 NAME=face5e-5 bin/siat4080 src/20241025/train.py -lr 5e-3 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4
