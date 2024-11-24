# version 89249
bin/siatv100v2 src/20241104/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89250
bin/siatv100v2 src/20241104/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89251
bin/siatv100v2 src/20241104/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89252
bin/siatv100v2 src/20241104/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 

# version 89755 1e-4cltr0.1 (NAN at epoch 1)

# version 89760 1e-4cltr0.05 (NAN at epoch 1)


# version 89741  scrath 1e-5

#####################################
# We found that all above version is in corrected because it forget to set_light_direction during training.
#####################################

#version_90367
bin/siatv100 src/20241104/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

#version_90374
bin/siatv100 src/20241104/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

#version_90375
bin/siatv100 src/20241104/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

#version_90376
bin/siatv100 src/20241104/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1


########
# memorization test 
########

bin/siatv100 src/20241104/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1 -dataset_val_split "/data/pakkapon/datasets/face/face60k/train-single.json" -dataset_train_split "/data/pakkapon/datasets/face/face60k/train-single.json" --dataset_train_multiplier 5000

bin/siatv100 src/20241104/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1 -dataset_val_split "/data/pakkapon/datasets/face/face60k/train-single.json" -dataset_train_split "/data/pakkapon/datasets/face/face60k/train-single.json"  --dataset_train_multiplier 5000
