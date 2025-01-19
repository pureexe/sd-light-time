# Experiment 1: background jitter withclip/without clip on multiple scenes
# Experiment 1.1: color jitter withclip multiple scene  # version_96220
bin/siatv100 src/20250115_triplet_loss/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_color_jitter_defareli \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff_conv_v2"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    --triplet_background

# color jitter noclip multiple scene version_96221

bin/siatv100 src/20250115_triplet_loss/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_color_jitter \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff_conv_v2"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    --triplet_background

# Experiment 2: False shading

# version_96227

bin/siatv100 src/20250115_triplet_loss/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_color_jitter_defareli \
    --batch_size 4 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff_conv_v2"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    --triplet_background \
    --false_shading

# version_96228
bin/siatv100 src/20250115_triplet_loss/train.py \
    -lr 1e-4 \
    --guidance_scale 1.0 \
    --network_type sd_color_jitter \
    --batch_size 8 \
    -c 1 \
    --feature_type clip \
    --shadings_dir "control_shading_from_ldr27coeff_conv_v2"\
    --backgrounds_dir "images"\
    -dataset "/data/pakkapon/datasets/multi_illumination/spherical/train" \
    -dataset_val "/data/pakkapon/datasets/multi_illumination/spherical/val" \
    -dataset_val_split "/data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json" \
    --triplet_background \
    --false_shading
