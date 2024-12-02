# static primpt 
cd /ist/ist-share/vision/pakkapon/relight/sd-light-time

# version_87186
bin/siatv100 src/20241027/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1

#  version_87187
bin/siatv100 src/20241027/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1

# version_87188
bin/siatv100 src/20241027/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1

# version_87189
bin/siatv100 src/20241027/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1


##### resume training

# LR 1e-4: version 88057 continue from version_87186
bin/siatv100 src/20241027/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_87186/checkpoints/epoch=000038.ckpt

# LR 5e-5: version 88058 continue from version_87187
bin/siatv100 src/20241027/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_87187/checkpoints/epoch=000039.ckpt

# LR 5e-5: version 88059 continue from version_87188
bin/siatv100 src/20241027/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_87188/checkpoints/epoch=000039.ckpt

# LR 5e-4: version 88089 continue from version_87188
bin/siatv100 src/20241027/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_87189/checkpoints/epoch=000039.ckpt


### RESUME TRAINING 2 

# LR 1e-4: version 91518  continue from version_88057
bin/siatv100 src/20241027/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_88057/checkpoints/epoch=000078.ckpt

# LR 5e-5: version 91519 continue from version_88058
bin/siatv100 src/20241027/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_88058/checkpoints/epoch=000079.ckpt

# LR 1e-5: version  continue from version_88059
bin/siatv100 src/20241027/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_88059/checkpoints/epoch=000079.ckpt

# LR 1e-5: version  continue from version_88059
bin/siatv100 src/20241027/train.py -lr 5e-4 -ct no_control --feature_type shcoeff_fuse --guidance_scale 1.0 --batch_size 16 -c 1 -ckpt output/20241027/multi_mlp_fit/lightning_logs/version_88060/checkpoints/epoch=000079.ckpt

