# at epoch 79
#val depth
#output/20240918/multi_mlp_fit/lightning_logs/version_13/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 13
# val depth shcoeff
#output/20240918/multi_mlp_fit/lightning_logs/version_25/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 25
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 1.0,3.0 -c 79 -i 25
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 5.0,7.0 -c 79 -i 25


# val bothbae 
#output/20240918/multi_mlp_fit/lightning_logs/version_15/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=1 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 15
# val bothbae shcoeff
# output/20240918/multi_mlp_fit/lightning_logs/version_26/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 26
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 1.0,3.0 -c 79 -i 26
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 5.0,7.0 -c 79 -i 26



# val bae 
#output/20240918/multi_mlp_fit/lightning_logs/version_14/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=2 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 14
# output/20240918/multi_mlp_fit/lightning_logs/version_27/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 27
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 1.0,3.0 -c 79 -i 27
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/val_ddim.py -m multillum_test2 -g 5.0,7.0 -c 79 -i 27



# val no_control
#output/20240918/multi_mlp_fit/lightning_logs/version_16/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=3 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 16
# val no_control_shcoeff
#output/20240918/multi_mlp_fit/lightning_logs/version_24/checkpoints/epoch=000079.ckpt
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/val_ddim.py -m multillum_train,multillum_test -g 1.0,3.0,5.0,7.0 -c 79 -i 24
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/val_ddim.py -m multillum_test -g 1.0,3.0 -c 79 -i 24
CUDA_VISIBLE_DEVICES=0 bin/py src/20240918/val_ddim.py -m multillum_test -g 5.0,7.0 -c 79 -i 24
