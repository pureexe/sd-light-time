# version: 89738
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 

# version: 89739 [loss nan]
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

# version: 89740
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 

# version: 89741
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 


bin/siatv100 src/20241108/train.py -lr 5e-6 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 




###########################################################################################################
#  version_89755 1e-4cltr0.1 (NAN at epoch 1)
bin/siatv100 src/20241108/train.py -lr 1e-4 --ctrlnet_lr 0.1 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

# version 89760 1e-4cltr0.05 (NAN at epoch 1)
bin/siatv100 src/20241108/train.py -lr 1e-4 --ctrlnet_lr 0.05 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 


# version 89961 5e-6
bin/siatv100 src/20241108/train.py -lr 5e-6 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

# version 89961 1e-6
bin/siatv100 src/20241108/train.py -lr 1e-6  --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

# version 89963
###########################################################################################################

# version_90499 (continue from 89738) model sd: 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 -ckpt  output/20241108/multi_mlp_fit/lightning_logs/version_89738/checkpoints/epoch=000023.ckpt

# version version_90500 (continue from 89740)  model sd: 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_89740/checkpoints/epoch=000024.ckpt

# version version_90501 (continue from 89741)  model scatch : 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1  -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_89741/checkpoints/epoch=000019.ckpt

#  version version_90502 (continue from 89961)  model scatch : 5e-6
bin/siatv100 src/20241108/train.py -lr 5e-6 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_89961/checkpoints/epoch=000019.ckpt

#####################################
# version_90532 SD but without agadn
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_without_adagn --batch_size 4 -c 1 

# version_90533
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_without_adagn --batch_size 4 -c 1 


# version_90535 ADAGAN (act as a control)

bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 

# version_90536 ADAGAN
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 

