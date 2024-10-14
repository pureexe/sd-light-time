
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multillum_test3 -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multillum_test3 -i 3
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012/val_ddim.py -m multillum_train2 -i 3


CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 25
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multillum_test3 -i 26
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 25
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multillum_train2 -i 26
