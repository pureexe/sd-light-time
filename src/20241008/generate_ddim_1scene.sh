# val test scene v14
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.05,0.1,0.15,0.2 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.05,0.1,0.15,0.2 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.05,0.1,0.15,0.2 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.05,0.1,0.15,0.2 -i 37

0.05,0.1,0.15,0.2
0.25,0.3,0.35,0.4
0.45,0.5,0.55,0.6
0.65,0.70,0.75,0.8,0.95,1.0


#v11
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s  0.25,0.3,0.35,0.4 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s  0.25,0.3,0.35,0.4 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s  0.25,0.3,0.35,0.4 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s  0.25,0.3,0.35,0.4 -i 37

#v14
CUDA_VISIBLE_DEVICES=0 bin/si src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.45,0.5,0.55,0.6 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.45,0.5,0.55,0.6 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.45,0.5,0.55,0.6 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.45,0.5,0.55,0.6 -i 37

#v1
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.70,0.75,0.8,0.95,1.0 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.70,0.75,0.8,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.70,0.75,0.8,0.95,1.0 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.70,0.75,0.8,0.95,1.0 -i 37

# v14
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  1,1.25,1.5,1.75 -s 0.75,0.8 -i 37
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2,2.25,2.5,2.75 -s  0.75,0.8 -i 37
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3,4,5,6,7 -s 0.75,0.8 -i 37


CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  1,1.25,1.5,1.75 -s 0.85,0.9,0.95,1.0 -i 37
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2,2.25,2.5,2.75 -s  0.85,0.9,0.95,1.0 -i 37

CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3,4,5,6,7 -s 0.85,0.9,0.95,1.0 -i 37

####

CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3,4,5,6,7 -s 1.0 -i 37
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2,2.25,2.5,2.75 -s 1.0 -i 37

CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  1,1.25 -s 1.0 -i 37


CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  1.5,1.75 -s 0.65,0.7 -i 37

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  6,7 -s 0.85,0.9,0.95 -i 37

CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  4,5 -s 0.85,0.9,0.95 -i 37

CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  5 -s 0.9 -i 37

CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  4 -s 1.0 -i 37

CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g  2.5 -s 1.0 -i 37

### version 35

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1.0,1.25 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1.5,1.75 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.0,2.25 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.5,2.75 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3,4 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35

CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 1.0 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.95 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.95-i 35
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.9 -i 35

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.85 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.8 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.75 -i 35
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7  -s 0.7 -i 35


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 36
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1.75,2,2.25  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 36
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75,3,  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 4,5,6,7  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 36

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1.75,2,2.25  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 33
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75,3,  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 33
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 4,5,6,7  -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0  -i 33






CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7 -s 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 -i 35


#######
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3  -s 0.65,0.7  -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3  -s 0.75,0.8  -i 33
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3  -s 0.85,0.9  -i 33
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 3  -s 0.95,1.0  -i 33


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75  -s 0.65,0.7  -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75  -s 0.75,0.8  -i 33
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75  -s 0.85,0.9  -i 33
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.75  -s 0.95,1.0  -i 33


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.5  -s 0.65,0.7  -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.5  -s 0.75,0.8  -i 33
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.5  -s 0.85,0.9  -i 33
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -m multillum_test1_4light_strength -g 2.5  -s 0.95,1.0  -i 33
