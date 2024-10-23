NODE=v21 GPU=0 NAME=val_chk39_1 bin/silocal src/20241020/val_ddim.py -g 1.0,3.0,5.0,7.0
NODE=v17 GPU=1 NAME=val_chk39_2 bin/silocal src/20241020/val_ddim.py -g 1.25,1.5,1.75
NODE=v17 GPU=2 NAME=val_chk39_3 bin/silocal src/20241020/val_ddim.py -g 2.0,2.25,2.5,2.75
NODE=v28 GPU=2 NAME=val_chk39_4 bin/siat4080 src/20241020/val_ddim.py -g 4.0,6.0
NODE=v28 GPU=1 NAME=val_chk39_5 bin/siat4080 src/20241020/val_ddim.py -g 5.0,1.75
NODE=v28 GPU=0 NAME=val_chk39_6 bin/siat4080 src/20241020/val_ddim.py -g 7.0,2.75

NODE=v27 GPU=2 NAME=val_chk39_7 bin/siat4080 src/20241020/val_ddim.py -g 2.75
NODE=v27 GPU=1 NAME=val_chk39_8 bin/siat4080 src/20241020/val_ddim.py -g 2.5
NODE=v27 GPU=0 NAME=val_chk39_9 bin/siat4080 src/20241020/val_ddim.py -g 2.25

# tvalidate on train iamge
NODE=v28 GPU=0 NAME=val_chk39_1 bin/siat4080 src/20241020/val_ddim.py -m multillum_train2_light4 -g 1.0,3.0,5.0,7.0
NODE=v28 GPU=1 NAME=val_chk39_2 bin/siat4080 src/20241020/val_ddim.py -m multillum_train2_light4 -g 1.25,1.5,1.75
NODE=v28 GPU=2 NAME=val_chk39_3 bin/siat4080 src/20241020/val_ddim.py -m multillum_train2_light4 -g 2.0,2.25,2.5,2.75
NODE=v27 GPU=2 NAME=val_chk39_4 bin/siat4080 src/20241020/val_ddim.py -m multillum_train2_light4 -g 4.0,6.0

# without training on train image
NODE=v27 GPU=0 NAME=val_chk39_5 bin/siat4080 src/20241020/val_ddim.py -c 0 -m multillum_train2_light4 -g 1.0,3.0,5.0,7.0
NODE=v27 GPU=1 NAME=val_chk39_6 bin/siat4080 src/20241020/val_ddim.py -c 0 -m multillum_train2_light4 -g 1.25,1.5,1.75
NODE=v26 GPU=0 NAME=val_chk39_7 bin/siat4080 src/20241020/val_ddim.py -c 0 -m multillum_train2_light4 -g 2.0,2.25,2.5,2.75
NODE=v26 GPU=1 NAME=val_chk39_8 bin/siat4080 src/20241020/val_ddim.py -c 0 -m multillum_train2_light4 -g 4.0,6.0,2.75

# training, revalidate
NODE=v26 GPU=2 NAME=val_chk39_9 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 1.0,3.0,5.0,7.0
NODE=v21 GPU=0 NAME=val_chk39_10 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 1.25,1.5,1.75
NODE=v17 GPU=1 NAME=val_chk39_11 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 2.0,2.25,2.5,2.75
NODE=v17 GPU=2 NAME=val_chk39_11 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 4.0,6.0,2.75
NODE=v19 GPU=1 NAME=val_chk39_13 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 7.0,5.0

NODE=v28 GPU=0 NAME=val_chk39_14 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 1.5
NODE=v28 GPU=1 NAME=val_chk39_15 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 1.75
NODE=v28 GPU=2 NAME=val_chk39_16 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 2.25

NODE=v27 GPU=0 NAME=val_chk39_17 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 2.5
NODE=v27 GPU=1 NAME=val_chk39_18 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 2.75
NODE=v27 GPU=2 NAME=val_chk39_19 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 3.0


NODE=v26 GPU=0 NAME=val_chk39_19 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 4.0
NODE=v26 GPU=1 NAME=val_chk39_19 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 39,0 -g 5.0

NODE=v26 GPU=0 NAME=val_chk39_20 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4 -g 1.0,1.25,1.5,1.75,2.0,2.5,2.75,3.0,4.0,5.0,6.0,7.0

NODE=v21 GPU=3 NAME=val_chk39_20 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 4 -g 1.5,1.75,2.0,2.5,2.75,3.0,4.0,5.0,6.0,7.0

NODE=v28 GPU=0 NAME=val_chk39_20 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 2.0
NODE=v28 GPU=1 NAME=val_chk39_21 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 2.25
NODE=v28 GPU=2 NAME=val_chk39_21 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 2.5
NODE=v27 GPU=0 NAME=val_chk39_21 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 2.75
NODE=v27 GPU=1 NAME=val_chk39_22 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 3.0
NODE=v27 GPU=2 NAME=val_chk39_22 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 1.75
NODE=v26 GPU=1 NAME=val_chk39_23 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 1.5
NODE=v26 GPU=2 NAME=val_chk39_24 bin/siat4080 src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 1.25
NODE=v21 GPU=1 NAME=val_chk39_24 bin/silocal src/20241020/val_ddim.py -m multillum_test10_light4 -c 4,9,14,19,24,29,34 -g 1.0


