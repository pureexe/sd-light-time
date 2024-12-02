import os 
import socket

FOLDER_NAME = "20241127_ballonfly"
OUTPUT = f"output/{FOLDER_NAME}/"
OUTPUT_DIR = OUTPUT
OUTPUT_CHROMEBALL_DIR = f"{OUTPUT_DIR}/chrome_ball_sd"
OUTPUT_CHROMEBALL_SDXL_DIR = f"{OUTPUT_DIR}/chrome_ball_sdxl"
SRC = f"src/{FOLDER_NAME}/"
DATASET_ROOT_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train"
DATASET_VAL_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val"
DATASET_VAL_SPLIT = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"
OUTPUT_MULTI_MANUAL = f"output/{FOLDER_NAME}/multi_fit_manual"
OUTPUT_SINGLE = f"output/{FOLDER_NAME}/single_fit/"
