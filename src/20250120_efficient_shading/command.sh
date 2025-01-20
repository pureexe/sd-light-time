# Experiment 1:  Single scnee #version_96433
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# experiment 2 multiple scene

# experiment 2.1 version_96434

bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

# experiment 2.2 LR 1e-5 version_96453
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

# experiment 2.2 LR 1e-5 version_96457
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 
