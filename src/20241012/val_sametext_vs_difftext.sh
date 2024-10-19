CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 1,1.25,1.5,1.75  

CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 6.0

CUDA_VISIBLE_DEVICES=2 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 5.0

CUDA_VISIBLE_DEVICES=3 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 2.75

CUDA_VISIBLE_DEVICES=0 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 2.5

CUDA_VISIBLE_DEVICES=1 bin/py src/20241012/val_ddim.py -m multiillum_test30_light4 -g 1.75