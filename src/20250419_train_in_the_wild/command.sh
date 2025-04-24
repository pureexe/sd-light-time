#################################
# train normal map to make sure thing is work fine
# 106314
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_only_shading \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "normal_lotus"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/fit_normal.json" \
        --val_check_interval 0.2 \
        -specific_prompt _
# 106315
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_only_shading \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "normal_lotus"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/150k.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/fit_normal.json" \
        --val_check_interval 0.2 \
        -specific_prompt _

####
# Marigold with prompt
#### 
# version_106625
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 \
        -specific_prompt _
# version_106626
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 \
        -specific_prompt _
# version_106627
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 \
        -specific_prompt _
# version_106628
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 \
        -specific_prompt _

####
# Marigold NO prompt
#### 
# version_106629
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 5e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 
# version_106630
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-4 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 
# version_106631
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 5e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 
# version_106632
bin/siatv100 src/20250419_train_in_the_wild/train.py \
        -lr 1e-5 \
        --guidance_scale 1.0 \
        --network_type sd_no_bg \
        --batch_size 8 \
        -c 1 \
        --feature_type clip \
        --shadings_dir "shadings_marigold"\
        --backgrounds_dir "images" \
        --images_dir "images" \
        -dataset "/data/pakkapon/datasets/laion-shading/v3/train" \
        -dataset_split "/data/pakkapon/datasets/laion-shading/v3/train/index/100k_marigold.json"  \
        -dataset_val "/data/pakkapon/datasets/laion-shading/v3/test" \
        -dataset_val_split "/data/pakkapon/datasets/laion-shading/v3/test/index/main_v2.json" \
        --val_check_interval 0.2 
