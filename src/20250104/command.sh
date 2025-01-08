# Experiment 1: test training on 1 scene first
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
# Experiment 2 - train on 
#version_95349 #version_95308
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

#version_95350 version_95309
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

#version_95352 version_95310
    bin/siatv100 src/20250104/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip_shcoeff \
        --shadings_dir "control_shading_from_ldr27coeff"\
        --backgrounds_dir "control_shading_from_ldr27coeff"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

#version_95354/version_95311
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

####
# Experiment 2.1 resume model

# from version_95349

bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250104/multi_mlp_fit/lightning_logs/version_95349/checkpoints/epoch=000004.ckpt

#version_95350 
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250104/multi_mlp_fit/lightning_logs/version_95350/checkpoints/epoch=000004.ckpt

#version_95352 version_95310
    bin/siatv100 src/20250104/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip_shcoeff \
        --shadings_dir "control_shading_from_ldr27coeff"\
        --backgrounds_dir "control_shading_from_ldr27coeff"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
        -ckpt output/20250104/multi_mlp_fit/lightning_logs/version_95352/checkpoints/epoch=000003.ckpt

#version_95354
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -ckpt output/20250104/multi_mlp_fit/lightning_logs/version_95354/checkpoints/epoch=000002.ckpt



#################################################################
# Hypothesis: We may need to bring back the prompt for good starting 
# version_95381
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -specific_prompt "_" \
    -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250104/multi_mlp_fit/lightning_logs/version_95381/checkpoints/epoch=000000.ckpt
#version_95382
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip_shcoeff \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -specific_prompt "_"

#version_95385
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip_shcoeff \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -specific_prompt "_"

#version_95386
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    -specific_prompt "_"
#####
# TEST RUN 
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae_shcoeff \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/10scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

#
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type vae \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/10scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 
#
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip_shcoeff \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/10scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_no_bg \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --dataset_train_multiplier 100 \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/10scenes_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 


###### 
# Experiment 3 No clip
# In this experiment, we will has another 2 test set which is no clip on single scene and no clip on multi-illuminatio

# version_95498 only shading, multi scene
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_only_shading \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" 

# version_95530 only shading, single scene
bin/siatv100 src/20250104/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_only_shading \
    --dataset_train_multiplier 1000 \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff"\
    --backgrounds_dir "control_shading_from_ldr27coeff"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_test.json"