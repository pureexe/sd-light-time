NODE=v16 GPU=0 NAME=val_depth bin/silocal src/20241021/val_ddim.py -m "multillum_train,multillum_test" -i 0 -c 79 -g 1,3,5,7
NODE=v16 GPU=1 NAME=val_both_bae bin/silocal src/20241021/val_ddim.py -m "multillum_train,multillum_test" -i 1 -c 79 -g 1,3,5,7
NODE=v16 GPU=2 NAME=val_bae bin/silocal src/20241021/val_ddim.py -m "multillum_train,multillum_test" -i 2 -c 79 -g 1,3,5,7
NODE=v16 GPU=3 NAME=val_nocontrol bin/silocal src/20241021/val_ddim.py -m "multillum_train,multillum_test" -i 3 -c 79 -g 1,3,5,7


# generate 

NODE=v28 GPU=0 NAME=val0 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 0 -c 79 -g 1,3,5,7
NODE=v28 GPU=1 NAME=val1 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 1 -c 79 -g 1,3,5,7
NODE=v28 GPU=2 NAME=val2 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 2 -c 79 -g 1,3,5,7
NODE=v27 GPU=0 NAME=val3 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 3 -c 79 -g 1,3,5,7
NODE=v27 GPU=1 NAME=val4 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 0 -c 79 -g 5,7
NODE=v27 GPU=2 NAME=val5 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 1 -c 79 -g 5,7
NODE=v26 GPU=0 NAME=val6 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 2 -c 79 -g 5,7
NODE=v26 GPU=1 NAME=val7 bin/siat4080 src/20241021/val_ddim.py -m "multillum_test2" -i 3 -c 79 -g 5,7