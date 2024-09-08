import os
import numpy as np
import json 


DATASET_NAME = "val_face_left"
GUIDANCE_SCALES = ['1.0', '3.0', '5.0', '7.0']
LEARNING_RATES = ['5e-4', '1e-4', '5e-5', '1e-5']
CHECKPOINTS = [19, 39, 59, 79]

ROOT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/20240902/{}/{}/new_light_block/{}/chk{}/lightning_logs/version_0/"

def main():

    for learning_rate in LEARNING_RATES:
        for checkpoint in CHECKPOINTS:
            for guidance_scale in GUIDANCE_SCALES:
                face_dir = ROOT_DIR.format(DATASET_NAME, guidance_scale, learning_rate, checkpoint) + "/face_light"
                counting_file = f"{ROOT_DIR.format(DATASET_NAME, guidance_scale, learning_rate, checkpoint)}/counting.json"
                counting = {
                    'left': 0,
                    'right': 0,
                    'left_ids': [],
                    'right_ids': [],
                }
                for filename in sorted(os.listdir(face_dir)):
                    if filename.endswith(".npy"):
                        light = np.load(os.path.join(face_dir, filename))
                        light = convert_to_grayscale(light.transpose())
                        if light[1] > 0:
                            counting['right'] += 1
                            counting['right_ids'].append(filename)
                        else:
                            counting['left'] += 1
                            counting['left_ids'].append(filename)
                print(counting_file)
                with open(counting_file, "w") as f:
                    json.dump(counting, f, indent=4)
                              

def convert_to_grayscale( v):
        """convert RGB to grayscale

        Args:
            v (np.array): RGB in shape of [3,...]
        Returns:
            np.array: gray scale array in shape [...] (1 dimension less)
        """
        assert v.shape[0] == 3
        return 0.299*v[0] + 0.587*v[1] + 0.114*v[2]

if __name__ == "__main__":
    main()