cd /pure/t1/project/sd-light-time/src/20250618_light_crossattention_anti_shock

# version_

# version_7432
python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

# version_7450
python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

# version_7451
python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

# version_7452
python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256

# version_7453
python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

# version_7454
python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

# version_7455
python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

# version_7456
python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256



# version_117520
python train.py --network_type=irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

python train.py --network_type=irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

python train.py --network_type=irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

python train.py --network_type=irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256

python train.py --network_type=albedo_normal_depth_irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

python train.py --network_type=albedo_normal_depth_irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

python train.py --network_type=albedo_normal_depth_irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

python train.py --network_type=albedo_normal_depth_irradiant --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256



