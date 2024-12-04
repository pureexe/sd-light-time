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




###########################################################################################################

# 91539 version (continue from 90499 / 89738) model sd: 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 -ckpt  output/20241108/multi_mlp_fit/lightning_logs/version_90499/checkpoints/epoch=000042.ckpt

# 91542 version version_ (continue from 90500 / 89740)  model sd: 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90500/checkpoints/epoch=000043.ckpt

# 91541 version  (continue from 90501 / 89741)  model scatch : 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1  -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90501/checkpoints/epoch=000034.ckpt

# 91540 version version_90502 (continue from 89961)  model scatch : 5e-6
bin/siatv100 src/20241108/train.py -lr 5e-6 --guidance_scale 1.0 --network_type scrath --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90502/checkpoints/epoch=000034.ckpt

####
# 91557 from version_90535 ADAGAN (act as a control)

bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90535/checkpoints/epoch=000011.ckpt

# 91557 ADAGAN (RESTART)
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 


# 91593 ADAGAN lr 5e-6
bin/siatv100 src/20241108/train.py -lr 5e-6 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 

# 91594 ADAGAN  lr 1e-6
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 

###########################################################################################################

# 91783 NOBG 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_no_bg --batch_size 4 -c 1 

# 91784 NOBG 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_no_bg --batch_size 4 -c 1 

# 91785 NOBG 1e-6
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_no_bg --batch_size 4 -c 1 


#  91786 NOSHADING 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_no_shading --batch_size 4 -c 1 

# 91787 NOSHADING 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_no_shading --batch_size 4 -c 1 

# 91788 NOSHADING 1e-6
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_no_shading --batch_size 4 -c 1 


###################################################################################
# Retrain SHCOEFF since found the DDIM Dataset bug
####################################################################################

# version_91864 adagn1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type diffusion_face_shcoeff

# version_91865 adagn1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type diffusion_face_shcoeff

# version_91865 adagn1e-6
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type diffusion_face_shcoeff


##############################################
# version_91869 1e-4 clip adagn
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type clip_shcoeff

# version_91870 1e-5 clip adagn
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type clip_shcoeff

# version_91871 1e-6 clip adagn
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type clip_shcoeff


# version_91872 1e-4 clip sd
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 --feature_type clip

# version_91873 1e-5 clip sd
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 --feature_type clip

# version_91873 1e-6 clip sd
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 --feature_type clip

############################################
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type diffusion_face_shcoeff
