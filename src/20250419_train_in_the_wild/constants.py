import os 
import socket

FOLDER_NAME = "20250419_train_in_the_wild"
OUTPUT = f"output/{FOLDER_NAME}/" 
OUTPUT_DIR = OUTPUT
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/train"
DATASET_VAL_DIR = "/data/pakkapon/datasets/multi_illumination/spherical/train"
DATASET_VAL_SPLIT = "/data/pakkapon/datasets/multi_illumination/spherical/14n_copyroom10_test.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"
# OUTPUT_MULTI = f"output/{FOLDER_NAME}/storage_t1"
# OUTPUT_MULTI = f"/pure/t1/checkpoints/sd-light-time/{FOLDER_NAME}"

# /data/pakkapon/datasets/face/ffhq_defareli/viz