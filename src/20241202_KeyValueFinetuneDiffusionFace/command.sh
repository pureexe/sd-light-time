# DOME @v3 teng account

# diffusion key value  1e-4 ver 91829
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type diffusionface_keyvalue -lr 1e-4
# diffusion key value  1e-5 ver 91830
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type diffusionface_keyvalue -lr 1e-5
# diffusion key value  1e-6 ver 91831
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type diffusionface_keyvalue -lr 1e-6


# diffusion key value no_controlnet 1e-4 ver 91832 
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type without_controlnet_keyvalue -lr 1e-4
# diffusion key value no_controlnet 1e-5 ver 91833 
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type without_controlnet_keyvalue -lr 1e-5
# diffusion key value no_controlnet 1e-6 ver 91839 (DEAD DUE TO DISK SPACE)
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type without_controlnet_keyvalue -lr 1e-6


# resume diffusion key value no_controlnet 1e-6 from 91839
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type without_controlnet_keyvalue -lr 1e-6 -ckpt output/20241202_KeyValueFinetuneDiffusionFace/multi_mlp_fit/lightning_logs/version_91839/checkpoints/epoch=000003.ckpt

# sanity check
bin/siatv100 src/20241202_KeyValueFinetuneDiffusionFace/train.py --batch_size 4 --guidance_scale 1.0 --network_type diffusionface_keyvalue -lr 1e-6