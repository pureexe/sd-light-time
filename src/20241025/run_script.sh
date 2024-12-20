NODE=v23 GPU=0 NAME=face1e-4 bin/silocal src/20241025/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16

NODE=v26 GPU=3 NAME=face1e-5 bin/siat4080 src/20241025/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4

NODE=v27 GPU=3 NAME=face5e-5 bin/siat4080 src/20241025/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4

#NODE=v28 GPU=3 NAME=face5e-5 bin/siat4080 src/20241025/train.py -lr 5e-3 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4

#v3 continue from v0 1e-4
NODE=v23 GPU=0 NAME=face1e-4 bin/silocal src/20241025/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16 -ckpt output/20241025/multi_mlp_fit/lightning_logs/version_0/checkpoints/epoch=000019.ckpt





###################
NODE=v28 GPU=3 NAME=face5e-5 bin/siat4080 src/20241025/train.py -lr 5e-3 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 4


# static primpt 
cd /ist/ist-share/vision/pakkapon/relight/sd-light-time
bin/siatv100 src/20241025/train.py -lr 1e-4 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16 

bin/siatv100 src/20241025/train.py -lr 5e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16 

bin/siatv100 src/20241025/train.py -lr 1e-5 -ct no_control --feature_type shcoeff_order2 --guidance_scale 1.0 --batch_size 16 

# parser.add_argument('-specific_prompt', type=str, default="a photorealistic image") 
