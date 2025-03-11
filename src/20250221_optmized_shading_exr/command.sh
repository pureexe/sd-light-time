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

# version version_102269/101149/100429/99826 # incorrect run, this use rendered shading as input instead of actual shading. 
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101150/checkpoints/epoch=000146.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100429/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_99826/checkpoints/epoch=000050.ckpt


 # Experiment 2 - Run with new shading but also the image that shade from that shade 
# version_/101150/100430/99828
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


# Experiment 4A - REBOOT 101760/version_101153 (SINGLE LIGHT)version_101004
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101153/checkpoints/epoch=000048.ckpt


# Experiment 4B -  REBOOT 101761/version_101154 (SINGLE LIGHT)version_101005
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101154/checkpoints/epoch=000049.ckpt

## TUNE LEARNING RATE  

# 102272/ version_101578 / version_101105
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101578/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101105/checkpoints/epoch=000048.ckpt

# 102273/ version_101579/version_101106
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101579/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101106/checkpoints/epoch=000048.ckpt

# 102274/version_101580/version_101107
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101580/checkpoints/epoch=000099.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101107/checkpoints/epoch=000048.ckpt

# 102279 version_101581/version_101108
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101581/checkpoints/epoch=000099.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101108/checkpoints/epoch=000049.ckpt


# DATASET POTION 10, 20, 50, 100, 200, 500
# 10 scene version_102350
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_10_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"

# 20 scene version_102351
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_20_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"

# 50 scene version_102352
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_50_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"

# 100 scene version_102353
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_100_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"

# 200 scene version_102354
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_200_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"

# 500 scene version_102355
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
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_500_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json"