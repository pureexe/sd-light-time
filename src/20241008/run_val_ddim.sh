CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 1,1.25,1.5,1.75
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 2,2.25,2.5,2.75
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 3,4,5,6,7

CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 7
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 5
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 4

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 2.75

CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 2.5

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 1.5

CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 1.75

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 4

CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 5

CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 7

CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim.py -m multiillum_test30_light4 -c 99 -i 25 -g 6