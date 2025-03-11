# Experiment 1A - 101762/101083 (albedo / old gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_divalbedo \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt /ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250303_exr_more_condition/multi_mlp_fit/lightning_logs/version_101083/checkpoints/epoch=000048.ckpt

# Experiment 1B 101763/ 101084 (albedo / new gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_divalbedo \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250303_exr_more_condition/multi_mlp_fit/lightning_logs/version_101084/checkpoints/epoch=000047.ckpt"

# Experiment 2A - 101764/101094 (edge / old gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250303_exr_more_condition/multi_mlp_fit/lightning_logs/version_101094/checkpoints/epoch=000048.ckpt"

# Experiment 2B  - 101765/101095  (edge / new gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" \
        -ckpt "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250303_exr_more_condition/multi_mlp_fit/lightning_logs/version_101095/checkpoints/epoch=000048.ckpt"


#######################
# CANCLE 3A and 3B, need data multipiler not -c 50

# # Experiment 3A - 97  (edge_singlelight / old gt)
# bin/siatv100 src/20250303_exr_more_condition/train.py \
#         -lr 1e-4 \
#         --guidance_scale 1.0 \
#         --network_type sd_shading_with_edge \
#         --batch_size 8 \
#         -c 50 \
#         --feature_type clip \
#         --shadings_dir "control_shading_from_fitting_v3_exr"\
#         --backgrounds_dir "images" \
#         --images_dir "images"\
#         -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
#         -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# # Experiment 3B  - 98 (edge_singlelight / new gt)
# bin/siatv100 src/20250303_exr_more_condition/train.py \
#         -lr 1e-4 \
#         --guidance_scale 1.0 \
#         --network_type sd_shading_with_edge \
#         --batch_size 8 \
#         -c 50 \
#         --feature_type clip \
#         --shadings_dir "control_shading_from_fitting_v3_exr"\
#         --backgrounds_dir "control_render_from_fitting_v2" \
#         --images_dir "control_render_from_fitting_v2"\
#         -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
#         -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# # Experiment 4A - 99 (edge_singlelight / old gt)
# bin/siatv100 src/20250303_exr_more_condition/train.py \
#         -lr 1e-4 \
#         --guidance_scale 1.0 \
#         --network_type sd_shading_with_edge \
#         --batch_size 8 \
#         -c 50 \
#         --feature_type clip \
#         --shadings_dir "control_shading_from_fitting_v3_exr"\
#         --backgrounds_dir "images" \
#         --images_dir "images"\
#         -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
#         -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# # Experiment 4B  - 100
# bin/siatv100 src/20250303_exr_more_condition/train.py \
#         -lr 1e-4 \
#         --guidance_scale 1.0 \
#         --network_type sd_shading_with_edge \
#         --batch_size 8 \
#         -c 50 \
#         --feature_type clip \
#         --shadings_dir "control_shading_from_fitting_v3_exr"\
#         --backgrounds_dir "control_render_from_fitting_v2" \
#         --images_dir "control_render_from_fitting_v2"\
#         -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
#         -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
#         -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 3A - 101766  (albedo_singlelight / old gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_divalbedo \
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

# Experiment 3B  - 101767 (albedo_singlelight / new gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_divalbedo \
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

# Experiment 4A - 101768 (edge_singlelight / old gt)
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
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

# Experiment 4B  - 101769
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
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
