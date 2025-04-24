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

# version /101149/100429/99826 # incorrect run, this use rendered shading as input instead of actual shading. 
# version_102269 is an incorrect resume code
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
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101150/checkpoints/epoch=000146.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100429/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_99826/checkpoints/epoch=000050.ckpt


 # Experiment 2 - Run with new shading but also the image that shade from that shade 
# version_103212/102774/102510/101150/100430/99828
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102774/checkpoints/epoch=000224.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102510/checkpoints/epoch=000197.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101150/checkpoints/epoch=000146.ckpt
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_100430/checkpoints/epoch=000097.ckpt
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

# 103368/102776/102272/ version_101578 / version_101105
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102776/checkpoints/epoch=000176.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102272/checkpoints/epoch=000149.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101578/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101105/checkpoints/epoch=000048.ckpt

# 103369/102777/102273/ version_101579/version_101106
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102777/checkpoints/epoch=000176.ckpt
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102273/checkpoints/epoch=000149.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101579/checkpoints/epoch=000098.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101106/checkpoints/epoch=000048.ckpt

# 103370/102778/102274/version_101580/version_101107
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102778/checkpoints/epoch=000177.ckpt
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102274/checkpoints/epoch=000150.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101580/checkpoints/epoch=000099.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101107/checkpoints/epoch=000048.ckpt

# 103371/version_102773/version_102279 version_101581/version_101108
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
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102773/checkpoints/epoch=000177.ckpt       
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102279/checkpoints/epoch=000150.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101581/checkpoints/epoch=000099.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_101108/checkpoints/epoch=000049.ckpt


# DATASET POTION 10, 20, 50, 100, 200, 500
# 10 scene (REBOOT)  103351/102781/102371 | version_102350
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 100 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_10_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102781/checkpoints/epoch=000042.ckpt
        # -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102371/checkpoints/epoch=000030.ckpt

# 20 scene (REBOOT) 103352/102782/102372 | version_102351
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 50 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_20_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102782/checkpoints/epoch=000042.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102372/checkpoints/epoch=000030.ckpt

# 50 scene (REBOOT) 103353/102783/102373 | version_102352
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 20 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_50_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102783/checkpoints/epoch=000042.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102373/checkpoints/epoch=000030.ckpt

# 100 scene (REBOOT) 103354/102784/102374 | version_102353
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        --dataset_train_multiplier 10 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_100_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102784/checkpoints/epoch=000046.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102374/checkpoints/epoch=000030.ckpt

# 200 scene (REBOOT) 103355/102785/102375 | version_102354
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        --dataset_train_multiplier 5 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_200_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102785/checkpoints/epoch=000042.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102375/checkpoints/epoch=000030.ckpt

# 500 scene (REBOOT) 103356/102786/102376 | version_102355
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --dataset_train_multiplier 2 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/multi_500_scenes_all.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102786/checkpoints/epoch=000042.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102376/checkpoints/epoch=000030.ckpt


# face dataset #103211/#102771/#102427
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102771/checkpoints/epoch=000037.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102427/checkpoints/epoch=000024.ckpt

# verify Image size 256 on training will create the problem or not
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_256" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images_256"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json"

# face dataset #103288 / 102686 #pipeline SDFACE
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102686/checkpoints/epoch=000024.ckpt

# experiment face on with diffusionface_feature 103625
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd \
        --batch_size 8 \
        -c 1 \
        --feature_type diffusion_face \
        --shadings_dir "shadings" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json"

# experiment face on face256 with diffusion face feqature 
# 103748
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd \
        --batch_size 8 \
        -c 1 \
        --feature_type diffusion_face \
        --shadings_dir "shadings_256" \
        --backgrounds_dir "backgrounds_256" \
        --images_dir "images_256"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json"

# supect the learning rate cause problem 
# LR5e-5 103400/102803  -- (failed run 103208)
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102803/checkpoints/epoch=000011.ckpt

# LR1e-5 103401/102804 -- (failed run 103209)
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings" \
        --backgrounds_dir "backgrounds" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/face/ffhq_defareli/train" \
        -dataset_val "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial" \
        -dataset_val_split "/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102804/checkpoints/epoch=000011.ckpt

# Don't worry. render v2 is compatible with shading v3 


# decay learning rate to 1e-5 103286/103072/version_102595 (previously version_102510/101150/100430/99828)
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_103072/checkpoints/epoch=000032.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102595/checkpoints/epoch=000031.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102510/checkpoints/epoch=000160.ckpt \
        #--restart_ckpt 1

# decay learning rate to 1e-5 version_103073/version_102596 (previously version_102510/101150/100430/99828)
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all_4lights.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102596/checkpoints/epoch=000031.ckpt
        #-ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_102510/checkpoints/epoch=000160.ckpt \
        #--restart_ckpt 1

#################################
# MORE WILDER DATASET
# 105180 / version_104458
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_104458/checkpoints/epoch=000014.ckpt
        
# 105181/version_104459
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_104459/checkpoints/epoch=000014.ckpt

# 105182/version_104460
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_104460/checkpoints/epoch=000014.ckpt


# version_104461
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" 


# TEST RUN  (1e-4) version_104462
CUDA_VISIBLE_DEVICES=0 bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 4 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" 

# TEST RUN  (5e-4) version_104463
CUDA_VISIBLE_DEVICES=1 bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 4 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v2/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v2/train/index/main_v1.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v2/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v2/test/index/main_v1.json" 

###########################################
# NEW SHADING TEST
###########################################

# 105177
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shading_exr_perspective_fov_order6"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_105177/checkpoints/epoch=000017.ckpt

# 105178
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shading_exr_perspective_fov_order6"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_105178/checkpoints/epoch=000017.ckpt


# 105179
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shading_exr_perspective_fov_order6"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs/version_105179/checkpoints/epoch=000018.ckpt

#################################
# MORE WILDER DATASET v3
# 106222
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106223
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106224
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106225
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"

#################################
# MORE WILDER DATASET v3 - with prompt
# 106259
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106260
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106261
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"
# 106262
bin/siatv100 src/20250221_optmized_shading_exr/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json"