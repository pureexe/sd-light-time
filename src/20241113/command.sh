#version_90439
bin/siatv100 src/20241113/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

#version_90440
bin/siatv100 src/20241113/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

bin/siatv100 src/20241113/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

bin/siatv100 src/20241113/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

########
# memorization test 
########

bin/siatv100 src/20241113/train.py -lr 1e-4 -ct no_control  --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1 -dataset_val_split "/data/pakkapon/datasets/face/face60k/train-single.json" -dataset_train_split "/data/pakkapon/datasets/face/face60k/train-single.json" --dataset_train_multiplier 5000


bin/siatv100 src/20241113/train.py -lr 1e-5 -ct no_control  --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1 -dataset_val_split "/data/pakkapon/datasets/face/face60k/train-single.json" -dataset_train_split "/data/pakkapon/datasets/face/face60k/train-single.json" --dataset_train_multiplier 5000
