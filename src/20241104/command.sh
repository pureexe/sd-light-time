# version 89249
bin/siatv100v2 src/20241104/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89250
bin/siatv100v2 src/20241104/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89251
bin/siatv100v2 src/20241104/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1

# version 89252
bin/siatv100v2 src/20241104/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 3.5 --batch_size 1 -c 1
