import os 
import socket

FOLDER_NAME = "20240925"
OUTPUT = f"output/{FOLDER_NAME}/"
OUTPUT_DIR = OUTPUT
OUTPUT_CHROMEBALL_DIR = f"{OUTPUT_DIR}/chrome_ball_sd"
OUTPUT_CHROMEBALL_SDXL_DIR = f"{OUTPUT_DIR}/chrome_ball_sdxl"
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/data/pakkapon/datasets/face/face2000"
DATASET_VAL_DIR = "/data/pakkapon/datasets/face/face2000"
DATASET_VAL_SPLIT = "/data/pakkapon/datasets/face/face2000/split-x-val-array.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"
OUTPUT_MULTI_MANUAL = f"output/{FOLDER_NAME}/multi_fit_manual"
OUTPUT_SINGLE = f"output/{FOLDER_NAME}/single_fit/"