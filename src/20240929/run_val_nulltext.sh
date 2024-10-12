CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 33 -c 254 -g 7

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 35 -c 349 -g 7

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 36 -c 304 -g 7

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 37 -c 299 -g 7


CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 33 -c 254 -g 5

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 35 -c 349 -g 5

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 36 -c 304 -g 5

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 37 -c 299 -g 5


CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 33 -c 254 -g 3

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 35 -c 349 -g 3

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 36 -c 304 -g 3

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 37 -c 299 -g 3

# NO_CONTROL
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 33 -c 254 -g 1

# BOTH_BAE
CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 35 -c 349 -g 1

#BAE
CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 36 -c 304 -g 1

# DEPTH
CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -i 37 -c 299 -g 1



# DEPTH, GUIDANCE SCALE = 7, 3, 2.5, 2 (v17,v17,v17,v5)

CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 37 -c 299 -g 7

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim.py -i 37 -c 299 -g 3

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -i 37 -c 299 -g 2.5

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim.py -i 37 -c 299 -g 2

# BOTH_BAE (v5, v5, v5, v8)
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 35 -c 349 -g 7

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim.py -i 35 -c 349 -g 3

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -i 35 -c 349 -g 2.5

CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 35 -c 349 -g 2

# BAE (v8, v8, v3, v3)
CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim.py -i 36 -c 304 -g 7

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -i 36 -c 304 -g 3

CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 36 -c 304 -g 2.5

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim.py -i 36 -c 304 -g 2

# NO CONTROL (v3, v3,v15, v16)
CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -i 33 -c 254 -g 7

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim.py -i 33 -c 254 -g 3

CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 33 -c 254 -g 2.5

CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 33 -c 254 -g 2

################## DDIM different type v7,v7,v22,v22

# depth
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_train2_relight -i 37 -c 299 -g 7,5,3,1

# bae 
CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_train2_relight -i 36 -c 304 -g 7,5,3,1

# both_bae
CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_train2_relight -i 35 -c 349 -g 7,5,3,1

# no_control
CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_train2_relight -i 33 -c 254 -g 7,5,3,1

################## NULL TEXT ON THOSE TRAIN 2 SCENE v13,v13,v16,v16

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -m multillum_train2_nulltext -i 37 -c 299 -g 7,3,2.5,2

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim.py -m multillum_train2_nulltext  -g 7,3,2.5,2 -i 35 -c 349

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -m multillum_train2_nulltext -i 36 -c 304 -g 7,3,2.5,2

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim.py -m multillum_train2_nulltext -i 33 -c 254 -g 7,3,2.5,2


# run the missing  v16,v16
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 35 -c 349 -g 1 --inversion_step 500,250

CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 35 -c 349 -g 1 --inversion_step 999,5,10,25,50,100,200

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 36 -c 304 -g 3 --inversion_step 999,5,10,25

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 36 -c 304 -g 3 50,100,200

CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 37 -c 299 -g 5 --inversion_step 200

CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 33 -c 254 -g 3 --inversion_step 200

# 999 rerun 
CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 36 -c 304 -g 3 --inversion_step 999,5,10,25


CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim_inversion_different_type_inverse.py -m multillum_ddim_bothway_guidance_val_array_v2 -i 35 -c 349 -g 1 --inversion_step 5,10,25,50,100,200,250



# RETURN 
## v11 (nocontrol 2.5) PREDICTING ## TMUX_ID17 # LAST CHECK 2147 
CUDA_VISIBLE_DEVICES=1 bin/py src/20240929/val_ddim.py -i 33 -c 254 -g 2.5

## v7 (bae 3) PREDICTING TMUX_ID4 ### LASTRUN 2205
CUDA_VISIBLE_DEVICES=2 bin/py src/20240929/val_ddim.py -i 36 -c 304 -g 3

## v11 (both_bae 3.0) TMUX_ID12 ### LAST CHECK 2205
CUDA_VISIBLE_DEVICES=3 bin/py src/20240929/val_ddim.py -m multillum_test_30_array_v2 -i 35 -c 349 -g 3


## v15 bothbae2 TMUXID_2, LAST CHECK 2204
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -m multillum_test_30_array_v2 -i 35 -c 349 -g 2  




# v7 (both_bae 2.0)###
CUDA_VISIBLE_DEVICES=0 bin/py src/20240929/val_ddim.py -i 35 -c 349 -g 2
