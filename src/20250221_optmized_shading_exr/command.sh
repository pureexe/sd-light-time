# Experiment 1 - Run with new shading but use source image  
# version 99826 # incorrect run, this use rendered shading as input instead of actual shading. 
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


# Experiment 2 - Run with new shading but also the image that shade from that shade 
# 99828
# Don't worry. render v2 is compatible with shading v3 

bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


# experiment 3, EXR that devide from image version  99998
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr_divide"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


# experiment FIT 1 SCENE first  #version_100235
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_train.json" \
        --dataset_train_multiplier 1000\
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 



 # 101006, 101007 is failed to resume
 
# EXPERIMENT 1 (RESUME)

# version version_101149/100429/99826 # incorrect run, this use rendered shading as input instead of actual shading. 
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100429/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_99826/checkpoints/epoch=000050.ckpt


 # Experiment 2 - Run with new shading but also the image that shade from that shade 
# version_101150/100430/99828
# Don't worry. render v2 is compatible with shading v3 

bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100430/checkpoints/epoch=000097.ckpt
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_99828/checkpoints/epoch=000050.ckpt

# experiment 3, EXR that devide from image version  101007/100429/99998
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr_divide"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100429/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_99998/checkpoints/epoch=000046.ckpt


# Experiment 4A - SINGLE LIGHT PER SCENE, REBOOT version_101153 (SINGLE LIGHT)version_101004
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 25 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


# Experiment 4B - Run with new shading but also the image that shade from that shade REBOOT version_101154 (SINGLE LIGHT)version_101005
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 25 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

## TUNE LEARNING RATE  

# version_101105
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# version_101106
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# version_101107
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-6 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"

# version_101108
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-6 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json"


