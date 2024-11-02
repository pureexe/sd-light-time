
NODE=v23 GPU=0 NAME=face1e-4 bin/silocal src/20241025/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16 -ckpt output/20241025/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000019.ckpt

# validation
# lef49
NODE=v15 GPU=0 NAME=left49g7 bin/sids src/20241025/val_ddim.py -m face_test_left -c 49 -g 7.0
NODE=v15 GPU=1 NAME=left49g5 bin/sids src/20241025/val_ddim.py -m face_test_left -c 49 -g 5.0
NODE=v15 GPU=2 NAME=left49g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 49 -g 3.0
NODE=v15 GPU=3 NAME=left49g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 49 -g 1.0

# right49
NODE=v12 GPU=0 NAME=right49g7 bin/sids src/20241025/val_ddim.py -m face_test_right -c 49 -g 7.0
NODE=v22 GPU=1 NAME=right49g5 bin/sids src/20241025/val_ddim.py -m face_test_right -c 49 -g 5.0
NODE=v22 GPU=2 NAME=right49g3 bin/sids src/20241025/val_ddim.py -m face_test_right -c 49 -g 3.0
NODE=v22 GPU=3 NAME=right49g1 bin/sids src/20241025/val_ddim.py -m face_test_right -c 49 -g 1.0


# lef39
NODE=v16 GPU=0 NAME=left39g7 bin/sids src/20241025/val_ddim.py -m face_test_left -c 39 -g 7.0
NODE=v16 GPU=1 NAME=left39g5 bin/sids src/20241025/val_ddim.py -m face_test_left -c 39 -g 5.0
NODE=v16 GPU=2 NAME=left39g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 39 -g 3.0
NODE=v16 GPU=3 NAME=left39g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 39 -g 1.0

# right39
NODE=v15 GPU=0 NAME=right39g7 bin/sids src/20241025/val_ddim.py -m face_test_right -c 39 -g 7.0
NODE=v15 GPU=1 NAME=right39g5 bin/sids src/20241025/val_ddim.py -m face_test_right -c 39 -g 5.0
NODE=v15 GPU=2 NAME=right39g3 bin/sids src/20241025/val_ddim.py -m face_test_right -c 39 -g 3.0
NODE=v15 GPU=3 NAME=right39g1 bin/sids src/20241025/val_ddim.py -m face_test_right -c 39 -g 1.0


# left29
NODE=v14 GPU=0 NAME=left29g7 bin/sids src/20241025/val_ddim.py -m face_test_left -c 29 -g 7.0
NODE=v14 GPU=1 NAME=left29g5 bin/sids src/20241025/val_ddim.py -m face_test_left -c 29 -g 5.0
NODE=v14 GPU=2 NAME=left29g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 29 -g 3.0
NODE=v14 GPU=3 NAME=left29g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 29 -g 1.0

# right29
NODE=v12 GPU=0 NAME=right29g7 bin/sids src/20241025/val_ddim.py -m face_test_right -c 29 -g 7.0
NODE=v12 GPU=1 NAME=right29g5 bin/sids src/20241025/val_ddim.py -m face_test_right -c 29 -g 5.0
NODE=v12 GPU=2 NAME=right29g3 bin/sids src/20241025/val_ddim.py -m face_test_right -c 29 -g 3.0
NODE=v12 GPU=3 NAME=right29g1 bin/sids src/20241025/val_ddim.py -m face_test_right -c 29 -g 1.0


# left19
NODE=v11 GPU=0 NAME=left19g7 bin/sids src/20241025/val_ddim.py -m face_test_left -c 19 -g 7.0
NODE=v11 GPU=1 NAME=left19g5 bin/sids src/20241025/val_ddim.py -m face_test_left -c 19 -g 5.0
NODE=v11 GPU=2 NAME=left19g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 19 -g 3.0
NODE=v11 GPU=3 NAME=left19g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 19 -g 1.0

# right19
NODE=v6 GPU=0 NAME=right29g7 bin/sids src/20241025/val_ddim.py -m face_test_right -c 19 -g 7.0
NODE=v6 GPU=1 NAME=right29g5 bin/sids src/20241025/val_ddim.py -m face_test_right -c 19 -g 5.0
NODE=v7 GPU=0 NAME=right29g3 bin/sids src/20241025/val_ddim.py -m face_test_right -c 19 -g 3.0
NODE=v7 GPU=1 NAME=right29g1 bin/sids src/20241025/val_ddim.py -m face_test_right -c 19 -g 1.0


# left9
NODE=v21 GPU=2 NAME=left9g7 bin/sids src/20241025/val_ddim.py -m face_test_left -c 9 -g 7.0
NODE=v21 GPU=3 NAME=left9g5 bin/sids src/20241025/val_ddim.py -m face_test_left -c 9 -g 5.0
NODE=v20 GPU=1 NAME=left9g3 bin/sids src/20241025/val_ddim.py -m face_test_left -c 9 -g 3.0
NODE=v19 GPU=0 NAME=left9g1 bin/sids src/20241025/val_ddim.py -m face_test_left -c 9 -g 1.0

#right9
NODE=v3 GPU=0 NAME=right9g7 bin/sids src/20241025/val_ddim.py -m face_test_right -c 9 -g 7.0
NODE=v3 GPU=2 NAME=right9g5 bin/sids src/20241025/val_ddim.py -m face_test_right -c 9 -g 5.0
NODE=v3 GPU=3 NAME=right9g3 bin/sids src/20241025/val_ddim.py -m face_test_right -c 9 -g 3.0
NODE=v8 GPU=1 NAME=right9g1 bin/sids src/20241025/val_ddim.py -m face_test_right -c 9 -g 1.0

################################################################
# Compute FaceLight score 
--saveLight 