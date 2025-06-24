cd /pure/t1/project/sd-light-time/src/20250619_crossattention_antishock_lr


# version_7487/7468 | version_7460 
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 -mul_lr_gate 1.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7468/checkpoints/epoch=000007.ckpt

# version_7488/7469 | version_7461
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 2.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7469/checkpoints/epoch=000007.ckpt

# version_7489/7470 | version_7462
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 5.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7470/checkpoints/epoch=000007.ckpt

# version_7490/7471 | version_7463
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 10.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7471/checkpoints/epoch=000007.ckpt

# version_7491/7472 | version_7464
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 50.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7472/checkpoints/epoch=000007.ckpt

# version_7492/7473 | version_7465
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 100.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7473/checkpoints/epoch=000007.ckpt

# version_7493/7474 | version_7466
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 500.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7474/checkpoints/epoch=000007.ckpt

# version_7494/7475 | version_7467
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 1000.0 --lora_rank 256 -ckpt /pure/t1/project/sd-light-time/output_t1/20250619_crossattention_antishock_lr/lightning_logs/version_7475/checkpoints/epoch=000007.ckpt

cd /pure/t1/project/sd-light-time/src/20250619_crossattention_antishock_lr
../../bin/shell
CUDA_VISIBLE_DEVICES=2 python val_ddim.py -m val -c 1,2,3,4 -i 7475


########################### 
# fade ratio 

# version_7526
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256 --fade_step 10000

# version_7527
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256 --fade_step 25000

# version_7528
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256 --fade_step 50000

# version_7529
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256 --fade_step 100000


