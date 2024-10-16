# version generate
NODE=v28 GPU=0 NAME=vae256 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=1 NAME=vae128 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae128 --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=2 NAME=vae64 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae64 --guidance_scale 3.0 --batch_size 8
NODE=v28 GPU=3 NAME=vae32 bin/siat4080 src/20241013/train.py -lr 1e-4 -ct depth --feature_type vae32 --guidance_scale 3.0 --batch_size 8


NODE=v27 GPU=0 NAME=val_test_vae256 bin/siat4080 src/20241013/val_ddim.py -m multillum_test3 -i 0 
NODE=v27 GPU=1 NAME=val_test_vae128 bin/siat4080 src/20241013/val_ddim.py -m multillum_test3 -i 1
NODE=v27 GPU=2 NAME=val_test_vae64 bin/siat4080 src/20241013/val_ddim.py -m multillum_test3 -i 2
NODE=v27 GPU=3 NAME=val_test_vae32 bin/siat4080 src/20241013/val_ddim.py -m multillum_test3 -i 3


NODE=v26 GPU=0 NAME=val_train_vae256 bin/siat4080 src/20241013/val_ddim.py -m multillum_train2 -i 0 
NODE=v26 GPU=1 NAME=val_train_vae128 bin/siat4080 src/20241013/val_ddim.py -m multillum_train2 -i 1
NODE=v26 GPU=2 NAME=val_train_vae64 bin/siat4080 src/20241013/val_ddim.py -m multillum_train2 -i 2
NODE=v26 GPU=3 NAME=val_train_vae32 bin/siat4080 src/20241013/val_ddim.py -m multillum_train2 -i 3