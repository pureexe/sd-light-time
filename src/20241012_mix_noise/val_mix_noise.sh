
# v12
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.0' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.1' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.2' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.3' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2

CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.4' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.5' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.6' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.7' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2

CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.8' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.9' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '1.0' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2

#v16
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.0' -g '3.0,4.0,5.0,6.0,7.0' -i 2
#v21
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.1' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.2' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.3' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.4' -g '3.0,4.0,5.0,6.0,7.0' -i 2

#v22
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.5' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.6' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.7,0.8' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.9,1.0' -g '3.0,4.0,5.0,6.0,7.0' -i 2


CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -s '0.5' -g '3.0' -i 2

#########################################################################
#v12
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '0.5' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '1.0' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '0.5' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '1.0' -g '1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2

#v15
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '0.5' -g '2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '1.0' -g '2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '0.5' -g '2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '1.0' -g '2.0,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0' -i 2

#v16
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '0.5' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '1.0' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '0.5' -g '3.0,4.0,5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '1.0' -g '3.0,4.0,5.0,6.0,7.0' -i 2

#v21
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '0.5' -g '5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_train2_light4" -s '1.0' -g '5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '0.5' -g '5.0,6.0,7.0' -i 2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_mix_noise/val_ddim_mix.py -m "multillum_test10_light4" -s '1.0' -g '5.0,6.0,7.0' -i 2