import os 
import socket

FOLDER_NAME = "20250104"
OUTPUT = f"output/{FOLDER_NAME}/"
OUTPUT_DIR = OUTPUT
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/train"
DATASET_VAL_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/train"
DATASET_VAL_SPLIT = "/data/pakkapon/datasets/multi_illumination/spherical/14n_copyroom10_test.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"

# /data/pakkapon/datasets/face/ffhq_defareli/viz