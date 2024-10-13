# val test scene v14
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -g 1,2 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -g 1,2 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -g 1,2 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -g 1,2 -i 37

#v11
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -g 1.5,2.5 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -g 1.5,2.5 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -g 1.5,2.5 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -g 1.5,2.5 -i 37


#v3
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -g 3,3.5 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -g 3,3.5 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -g 3,3.5 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -g 3,3.5 -i 37

#v1
CUDA_VISIBLE_DEVICES=0 bin/py src/20241008/val_ddim_strength.py -g 4,4.5 -i 33
CUDA_VISIBLE_DEVICES=1 bin/py src/20241008/val_ddim_strength.py -g 4,4.5 -i 35
CUDA_VISIBLE_DEVICES=2 bin/py src/20241008/val_ddim_strength.py -g 4,4.5 -i 36
CUDA_VISIBLE_DEVICES=3 bin/py src/20241008/val_ddim_strength.py -g 4,4.5 -i 37

