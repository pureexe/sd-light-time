import os 
import socket

FOLDER_NAME = "20250619_crossattention_antishock_lr"
OUTPUT = f"output_t1/{FOLDER_NAME}/" 
OUTPUT_DIR = OUTPUT
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/train"
DATASET_VAL_DIR = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/val"
DATASET_VAL_SPLIT = "/pure/f1/datasets/multi_illumination/diffusionrenderer/v1/index/val.json"
OUTPUT_MULTI = f"/pure/t1/checkpoints/sd-light-time/{FOLDER_NAME}"

