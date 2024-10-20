# version generate
#0
NODE=v23 GPU=1 NAME=deepfloyd bin/siat src/20241020/train.py -lr 1e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 12

# 5e-4 version_86524
cd /ist/ist-share/vision/pakkapon/relight/sd-light-time
bin/siatv100 src/20241020/train.py -lr 5e-4 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8 

# 5e-5 version_86527
cd /ist/ist-share/vision/pakkapon/relight/sd-light-time
bin/siatv100 src/20241020/train.py -lr 5e-5 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8 

# 1e-5 version_86528
cd /ist/ist-share/vision/pakkapon/relight/sd-light-time
bin/siatv100 src/20241020/train.py -lr 1e-5 -ct deepfloyd --feature_type vae --guidance_scale 1.0 --batch_size 8 
