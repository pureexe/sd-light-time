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


#  version_89755 1e-4cltr0.1 (NAN at epoch 1)
bin/siatv100 src/20241108/train.py -lr 1e-4 --ctrlnet_lr 0.1 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

# version 89760 1e-4cltr0.05 (NAN at epoch 1)
bin/siatv100 src/20241108/train.py -lr 1e-4 --ctrlnet_lr 0.05 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 

