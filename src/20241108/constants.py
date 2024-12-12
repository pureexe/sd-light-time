import os 
import socket

FOLDER_NAME = "20241108"
OUTPUT = f"output/{FOLDER_NAME}/"
OUTPUT_DIR = OUTPUT
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/data/pakkapon/datasets/face/ffhq_defareli/train"
DATASET_VAL_DIR = "/data/pakkapon/datasets/face/ffhq_defareli/viz"
DATASET_VAL_SPLIT = "/data/pakkapon/datasets/face/ffhq_defareli/viz/val-viz-array.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"

# /data/pakkapon/datasets/face/ffhq_defareli/viz