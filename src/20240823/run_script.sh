#version 0
bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under -lr 1e-4

#version 1
bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under -lr 1e-5

#version 2
bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_norm -lr 1e-4

#version 3
bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_norm -lr 1e-5

#############################################################################
#!!!!!!!!!!!!!!!!
#version 4 
bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 1e-4
#!!!!!!!!!!!!!!!!
#version 5 
CUDA_VISIBLE_DEVICES=1 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 1e-5

#version 6
#CUDA_VISIBLE_DEVICES=2 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_norm --batch_size 4 -lr 1e-4

#version 7
#CUDA_VISIBLE_DEVICES=3 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_norm --batch_size 4 -lr 1e-5

#!!!!!!!!!!!!!!!!
#version 8 #CRASH 
CUDA_VISIBLE_DEVICES=0 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-4

#version 9 #CRASH #MOVE TO V4
#CUDA_VISIBLE_DEVICES=1 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 1e-4 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_4/checkpoints/epoch=000018.ckpt

#!!!!!!!!!!!!!!!!
#version 10 #CRASH
CUDA_VISIBLE_DEVICES=2 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-5

#version 11 #CRASH
CUDA_VISIBLE_DEVICES=3 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 1e-5 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_5/checkpoints/epoch=000018.ckpt

#version 12 #CRASH
CUDA_VISIBLE_DEVICES=0 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-4 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_8/checkpoints/epoch=000006.ckpt

#version 13 #CRASH, SHOULD CANCLE
CUDA_VISIBLE_DEVICES=2 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-5 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_8/checkpoints/epoch=000006.ckpt #OH NO WRONG CHECKPOINT LOAD HERE

#version 14
CUDA_VISIBLE_DEVICES=0 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-4 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_12/checkpoints/epoch=000015.ckpt

#version 15
CUDA_VISIBLE_DEVICES=2 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 5e-5 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_10/checkpoints/epoch=000006.ckpt #OH NO WRONG CHECKPOINT LOAD HERE

#version 16
CUDA_VISIBLE_DEVICES=3 bin/py src/20240823/train.py -d /data/pakkapon/datasets/unsplash-lite/train_under --batch_size 4 -lr 1e-5 -ckpt output/20240823/multi_mlp_fit/lightning_logs/version_11/checkpoints/epoch=000035.ckpt

