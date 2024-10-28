

bin/siatv100 src/20241027/val_coeff3.py


# val 27, 10, 20 

# val coeff 27
bin/siatv100 src/20241027/val_coeff27.py -m faceval10k_fuse_test_right -c 10,20,0 -g 1.0,7.0,5.0,3.0
bin/siatv100 src/20241027/val_coeff27.py -m faceval10k_fuse_test_left -c 10,20,0 -g 1.0,7.0,5.0,3.0

# val coeff 3
bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_right -c 10,20,0 -g 1.0,7.0,5.0,3.0


# left 
bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_left -c 10,20,0 -g 1.0
bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_left -c 10,20,0 -g 7.0
bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_left -c 10,20,0 -g 5.0
bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_left -c 10,20,0 -g 3.0

bin/siatv100 src/20241027/val_coeff3.py -m faceval10k_fuse_test_left -c 20 -g 1.0

