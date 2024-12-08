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

# version_91875 1e-5 clip sd
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 --feature_type clip

# version_91876 1e-6 clip sd
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd --batch_size 4 -c 1 --feature_type clip

##############################################
#version 92037
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type shcoeff_order2

#version 92047
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type shcoeff_order2

# version 92049 teng
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_only_adagn --batch_size 4 -c 1 --feature_type shcoeff_order2


###################
# version 92205
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_only_shading  -c 1 --batch_size 4

# version 92206
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_only_shading  -c 1 --batch_size 4

# version 92207
bin/siatv100 src/20241108/train.py -lr 1e-6 --guidance_scale 1.0 --network_type sd_only_shading  -c 1 --batch_size 4


##################################### Continue training with controlnet part only
#f from version_90532 SD but without agadn
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_without_adagn --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90532/checkpoints/epoch=000008.ckpt

# version_90533
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type sd_without_adagn --batch_size 4 -c 1 -ckpt output/20241108/multi_mlp_fit/lightning_logs/version_90533/checkpoints/epoch=000008.ckpt

##################################### Train with inpainting condition
# version_92372 shading and masked background 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type inpaint --batch_size 4 -c 1

# shading and masked background 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type inpaint --batch_size 4 -c 1


### masked background only 1e-4
bin/siatv100 src/20241108/train.py -lr 1e-4 --guidance_scale 1.0 --network_type inpaint_no_shading --batch_size 4 -c 1

### version_92438 background only 1e-5
bin/siatv100 src/20241108/train.py -lr 1e-5 --guidance_scale 1.0 --network_type inpaint_no_shading --batch_size 4 -c 1




############################################
# after fixing the bug. let run valdiation on the lastest step to see if it change anything (yet)

# validation on 1e-4 
bin/siatv100 src/20241108/val_ddim.py -i 91864 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91865 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91866 -m valid2left,valid2right,train2left,train2right

bin/siatv100 src/20241108/val_ddim.py -i 91869 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91870 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91871 -m valid2left,valid2right,train2left,train2right

bin/siatv100 src/20241108/val_ddim.py -i 91872 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91875 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 91876 -m valid2left,valid2right,train2left,train2right

bin/siatv100 src/20241108/val_ddim.py -i 92037 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 92047 -m valid2left,valid2right,train2left,train2right
bin/siatv100 src/20241108/val_ddim.py -i 92049 -m valid2left,valid2right,train2left,train2right


# predict only shading 

bin/siatv100 src/20241108/val_ddim.py -i 92037 -m valid2left,train2left
bin/siatv100 src/20241108/val_ddim.py -i 92047 -m valid2left,train2left
bin/siatv100 src/20241108/val_ddim.py -i 92049 -m valid2left,train2left
bin/siatv100 src/20241108/val_ddim.py -i 92037 -m valid2right,train2right
bin/siatv100 src/20241108/val_ddim.py -i 92047 -m valid2right,train2right
bin/siatv100 src/20241108/val_ddim.py -i 92049 -m valid2right,train2right


