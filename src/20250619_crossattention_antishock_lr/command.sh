cd /pure/t1/project/sd-light-time/src/20250619_crossattention_antishock_lr


# version_7468 | version_7460 
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 -mul_lr_gate 1.0 --lora_rank 256

# version_7469 | version_7461
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 2.0 --lora_rank 256

# version_7470 | version_7462
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 5.0 --lora_rank 256

# version_7471 | version_7463
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 10.0 --lora_rank 256

# version_7472 | version_7464
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 50.0 --lora_rank 256

# version_7473 | version_7465
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 100.0 --lora_rank 256

# version_7474 | version_7466
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 500.0 --lora_rank 256

# version_7475 | version_7467
python train.py --network_type=normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4  -mul_lr_gate 1000.0 --lora_rank 256
