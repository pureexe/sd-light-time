import os 
import socket

FOLDER_NAME = "20241027"
OUTPUT = f"output/{FOLDER_NAME}/"
OUTPUT_DIR = OUTPUT
OUTPUT_CHROMEBALL_DIR = f"{OUTPUT_DIR}/chrome_ball_sd"
OUTPUT_CHROMEBALL_SDXL_DIR = f"{OUTPUT_DIR}/chrome_ball_sdxl"
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/data/pakkapon/datasets/face/face60k_fuse"
DATASET_VAL_DIR = "/data/pakkapon/datasets/face/face60k_fuse"
DATASET_VAL_SPLIT = "/data/pakkapon/datasets/face/face60k/train-viz-array.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"
OUTPUT_MULTI_MANUAL = f"output/{FOLDER_NAME}/multi_fit_manual"
OUTPUT_SINGLE = f"output/{FOLDER_NAME}/single_fit/"
