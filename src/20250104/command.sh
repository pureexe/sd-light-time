# test training on 1 scene first


# version 95208
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# version 95209
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# version 95211
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip_shcoeff \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# version 95212
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

##########################################
# version 95208
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"
