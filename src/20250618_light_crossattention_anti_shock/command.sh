cd /pure/t1/project/sd-light-time/src/20250618_light_crossattention_anti_shock

# version_

python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

python train.py --network_type=default --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256

python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 4

python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 16

python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 64

python train.py --network_type=albedo_normal_depth --batch_size=4 --grad_accum=8 --learning_rate=1e-4 --lora_rank 256




# version_7404/7398
python train.py --network_type=default --batch_size=2 --grad_accum=8 --learning_rate=5e-5 -ckpt /pure/t1/project/sd-light-time/output_t1/20250614_light_embed_condition/lightning_logs/version_7398/checkpoints/epoch=000003.ckpt

# version_7405/7399
python train.py --network_type=default --batch_size=2 --grad_accum=8 --learning_rate=1e-5 -ckpt /pure/t1/project/sd-light-time/output_t1/20250614_light_embed_condition/lightning_logs/version_7399/checkpoints/epoch=000003.ckpt

# version_7406/7400
python train.py --network_type=albedo_normal_depth --batch_size=2 --grad_accum=8 --learning_rate=1e-4 -ckpt /pure/t1/project/sd-light-time/output_t1/20250614_light_embed_condition/lightning_logs/version_7400/checkpoints/epoch=000001.ckpt

# version_7407/7401
python train.py --network_type=albedo_normal_depth --batch_size=2 --grad_accum=8 --learning_rate=5e-5 -ckpt /pure/t1/project/sd-light-time/output_t1/20250614_light_embed_condition/lightning_logs/version_7401/checkpoints/epoch=000001.ckpt

# version_7410/7402
python train.py --network_type=albedo_normal_depth --batch_size=2 --grad_accum=8 --learning_rate=1e-5 -ckpt /pure/t1/project/sd-light-time/output_t1/20250614_light_embed_condition/lightning_logs/version_7402/checkpoints/epoch=000001.ckpt