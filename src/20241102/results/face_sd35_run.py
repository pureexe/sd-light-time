import os 
import json 
from tqdm.auto import tqdm

def main():
    with open('/ist/ist-share/vision/relight/datasets/face/face60k/boy.json') as f:
        image_ids = json.load(f)
    
    for image_id in tqdm(image_ids):
        #for direction in ['left','right']:
        for direction in ['left']:
            PROMPT_FROM = "face of a boy"
            PROMPT_TO = f"face of a boy with sunlight illuminate on the {direction}"
            dir_id = image_id // 1000 * 1000
            image_path = f"/ist/ist-share/vision/relight/datasets/face/face60k/images/{dir_id:05d}/{image_id:05d}.jpg"

            output_dir = f"output/20241102_fluxdev/{direction}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{image_id:05d}.jpg")
            os.system(f"python src/20241102/face_flux_inversion.py --use_inversed_latents --image_path {image_path} --output_path {output_path}")
            exit()
if __name__ == "__main__":
    main()