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
# Experiment 3.1 2 scene #version_96636
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

# Experiment 3.2 5 scene  version_96637
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

# Experiment 3.3 10 scene version_96638
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

# experiemnt 3.4 20 scene version_96639
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

# experiemnt 3.5 50scene #version_96640
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

# experiemnt 3.6 96641
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


#######################################################################################
# Experiment 4 - Restart with coeff version 4 

# 97150
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97150/checkpoints/epoch=000000.ckpt

# 97151
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 200 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_5_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97151/checkpoints/epoch=000000.ckpt

# 97152
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_10_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97152/checkpoints/epoch=000000.ckpt

# 97153
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 50 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_20_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97153/checkpoints/epoch=000000.ckpt

# 97154
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 20 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_50_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97154/checkpoints/epoch=000000.ckpt

# 97155
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 10 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_100_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97155/checkpoints/epoch=000000.ckpt


#  97156
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v4"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v4"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97156/checkpoints/epoch=000000.ckpt


#######################################################################################
# Experiment 5 - Restart with coeff version 5

# 97221
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 1000 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# 97222
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 200 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_5_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# 97223
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_10_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# 97224
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 50 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_20_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# 97225
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 20 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_50_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# 97226
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 10 \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_100_scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


#  97227
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

##############################################################

# version_97357 (All back. too much learning rate) # THIS IS NAN
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 5e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"  

#version_97358
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 5e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"  
#version_97359
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"  

#version_97360
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 5e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

#version_97361
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"  

# version 97386 (continue from 97227) 
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97227/checkpoints/epoch=000049.ckpt

# version 97622 (continue from 97386/97227) 
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97622/checkpoints/epoch=000082.ckpt

## CONTINUE VERSION
# 97623 (from version_97358)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 5e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97623/checkpoints/epoch=000045.ckpt

# 97624 (version_97359)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-5 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97624/checkpoints/epoch=000045.ckpt

# 97625 (version_97360)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 5e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97625/checkpoints/epoch=000046.ckpt

#97626 (version_97361)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-6 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"  \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97626/checkpoints/epoch=000045.ckpt


# experiment 
# 3.1 new shading / new albedo version_97636
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97636/checkpoints/epoch=000012.ckpt
# 3.2 old shading / new albedo version_97637
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97637/checkpoints/epoch=000012.ckpt
# 3.3 new shading version_97638
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97638/checkpoints/epoch=000012.ckpt
######################################################
# version 98314 (continue from 97622/97386/97227) 
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97622/checkpoints/epoch=000083.ckpt

# 3.1 new shading / new albedo version_98315 (from version_97636)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97636/checkpoints/epoch=000028.ckpt

# 3.3 new shading version_98316 (version_97638)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_97638/checkpoints/epoch=000013.ckpt

##################################
# use shading from blender instead ๖

# version 98634 (98472)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_blender_mesh_othographic"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_98472/checkpoints/epoch=000050.ckpt

# version 98581
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_blender_mesh_perspective_v2"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"


############################
# 2025/02/10 Resume 
############################
# order2 version_98562  (continue from 98314/97622/97386/97227) 
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_hdr27coeff_conv_v5"\
    --backgrounds_dir "control_shading_from_hdr27coeff_conv_v5"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_98314/checkpoints/epoch=000134.ckpt

# 3.1 new shading / new albedo version_98563 (from version_98315/ version_97636)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_98315/checkpoints/epoch=000079.ckpt

# 3.3 new shading version_98564 (version_98316, version_97638)
bin/siatv100 src/20250120_efficient_shading/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_intrinsic_shading_diffuse"\
    --backgrounds_dir "control_intrinsic_albedo_shared"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs/version_98316/checkpoints/epoch=000064.ckpt
