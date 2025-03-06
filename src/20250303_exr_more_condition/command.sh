# Experiment 1A - 101083
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 


# Experiment 1B  101084
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 2A - 101094
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 2B  - 101095
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
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 3A - 97
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 50 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 3B  - 98
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 50 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 4A - 99
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 50 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "images" \
        --images_dir "images"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 

# Experiment 4B  - 100
bin/siatv100 src/20250303_exr_more_condition/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_shading_with_edge \
        --batch_size 8 \
        -c 50 \
        --feature_type clip \
        --shadings_dir "control_shading_from_fitting_v3_exr"\
        --backgrounds_dir "control_render_from_fitting_v2" \
        --images_dir "control_render_from_fitting_v2"\
        -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_split "/data/pakkapon/datasets/multi_illumination/spherical/index/singlelight_per_scene_train.json" \
        -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/train" \
        -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/index/14n_copyroom10_all.json" 
