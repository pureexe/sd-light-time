# Experiment 1:  Single scnee #version_96433 (This is incorrect, we need feature type clip)
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

# CLIP LR 1e-4 version 96458
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# CLIP LR 1e-5 #96461
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"

# CLIP LR 1e-6 #96462
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
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

# Experiment 3 - Train with differnet number of scene to see if it can relighting single scene
# Experiment 3.1 2 scene #version_96622
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 500 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_2_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# Experiment 3.2 5 scene  version_96623
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 200 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_5_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# Experiment 3.3 10 scene version_96625
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_10_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# experiemnt 3.4 20 scene version_96627
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 50 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_20_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# experiemnt 3.5 50scene #version_96629
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 20 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_50_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# experiemnt 3.6
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 10 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v3"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v3"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_100_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"