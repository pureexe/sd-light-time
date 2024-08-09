import os 
import socket

FOLDER_NAME = "20240801"
OUTPUT = f"output/{FOLDER_NAME}/"
SRC = f"src/{FOLDER_NAME}/"
#DATASET_ROOT_DIR = "/home/pakkapon/mnt_tl_vision17/pure/tu150/datasets/relight/face/ffhq"


DATASET_ROOT_DIR = "/data/pakkapon/datasets/shoe4light250"
#DATASET_ROOT_DIR = "/data/pakkapon/datasets/pointlight_shoe_rand100"
#DATASET_ROOT_DIR = "/data/pakkapon/datasets/pointlight_shoe_y_axis"
#DATASET_ROOT_DIR = "/data/pakkapon/datasets/synthetic2kyz"
VAL_Y_ROOT_DIR = "/data/pakkapon/datasets/synthetic2kyz_val_y"
VAL_Z_ROOT_DIR = "/data/pakkapon/datasets/synthetic2kyz_val_z"
#DATASET_ROOT_DIR = "/data/pakkapon/datasets/ffhq"
#DATASET_ROOT_DIR = "/pure/tu150/datasets/relight/face/ffhq"
OUTPUT_MULTI = f"output/{FOLDER_NAME}/multi_mlp_fit"
OUTPUT_MULTI_MANUAL = f"output/{FOLDER_NAME}/multi_fit_manual"
OUTPUT_SINGLE = f"output/{FOLDER_NAME}/single_fit/"

