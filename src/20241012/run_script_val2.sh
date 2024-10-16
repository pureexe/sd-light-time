
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multillum_test3 -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multillum_test3 -i 3
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 3


#CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 2 -g 4
#CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 3 -g 4
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 2 -g 3,4
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 3 -g 3,4

CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 2 -g 5,6,7
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 3 -g 5,6,7


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 26
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 25
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 26


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 2,2.25,2.5,2.75
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 26  -g 2,2.25,2.5,2.75
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 25 -g 2,2.25,2.5,2.75
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 26 -g 2,2.25,2.5,2.75


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 3,4,5,7
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 26  -g 3,4,5,7
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 25 -g 3,4,5,7
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 26 -g 3,4,5,7


CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 5,7
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 26  -g 5,7


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 1,1.25,1.5,1.75
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 2,2.25,2.5,2.75
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25 -g 3,4,5,7
