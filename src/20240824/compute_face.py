# val_grid is a validation at step 

from AffineConsistancy import AffineConsistancy
from UnsplashLiteDataset import UnsplashLiteDataset
import lightning as L
import torch
import argparse 
from LineNotify import LineNotify
import argparse
from constants import FOLDER_NAME
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--version", type=str, default="1")
parser.add_argument("-m", "--mode", type=str, default="human_left,human_right") 
parser.add_argument("-g", "--guidance_scale", type=str, default="1.0,3.0,5.0,7.0")
parser.add_argument("-c", "--checkpoint", type=str, default="20" )

args = parser.parse_args()
NAMES = ['new_light_block', 'new_light_block', 'new_light_block', 'new_light_block']
LRS = ['5e-4', '1e-4', '5e-5', '1e-5' ]

def main():
    versions = [int(a.strip()) for a in args.version.split(",")]
    guidance_scales = [float(a.strip()) for a in args.guidance_scale.split(",")]
    checkpoints = [int(a.strip()) for a in args.checkpoint.split(",")]
    modes = [a.strip() for a in args.mode.split(",")]
    os.chdir("thrid_party/DECA")
    for mode in modes:
        for version in versions:
                for checkpoint in checkpoints:
                     for guidance_scale in guidance_scales:
                        source_dir = f"../../output/{FOLDER_NAME}/val_{mode}/{guidance_scale}/{NAMES[version]}/{LRS[version]}/chk{checkpoint}/lightning_logs/version_0/"
                        input_dir = os.path.join(source_dir, "crop_image")
                        output_dir = os.path.join(source_dir, "face_ight")
                        cmd = f"deca-env/bin/python demos/demo_reconstruct.py -i {input_dir} -s {output_dir} --saveLight True"
                        print("================================")
                        print(output_dir)                        
                        print(cmd)
                        os.system(cmd)
 
                                
if __name__ == "__main__":
    main()