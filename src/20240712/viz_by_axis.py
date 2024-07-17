import os 
import json 
import skimage
import numpy as np 

def main():
    for axis in ['x','y','z']:
        for direction in ['minus', 'plus']:
            json_file = f"datasets/face/face2000_single/light_{axis}_{direction}.json"
            images = []
            with open(json_file, 'r') as f:
                data = json.load(f)
            for image_name in data:
                image_path = f'datasets/face/face2000_single_viz/ball_visualize_all/{image_name}.png'
                images.append(
                    skimage.io.imread(image_path)
                )
            images = np.concatenate(images, axis=0)
            skimage.io.imsave(f"datasets/face/face2000_single_viz//light_{axis}_{direction}.png", images)
            
                
if __name__ == "__main__":
    main()