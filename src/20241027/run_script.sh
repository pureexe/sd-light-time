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


