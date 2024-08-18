import os
import numpy as np
import json 


DATASET_NAME = "val_human_right"
GUIDANCE_SCALES = ['1.0', '3.0', '5.0', '7.0']
LEARNING_RATES = ['5e-4', '1e-4', '5e-5', '1e-5']
CHECKPOINTS = [0,1,2,3,4,5,10,15,20,25,30,35,40]

ROOT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/20240815/{}/{}/no_consistnacy/{}/chk{}/lightning_logs/version_0/"

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


    for version in range(4):
        #for step in [39900, 77900, 115900, 153900, 191900, 229900,267900, 305900, 343900, 381900, 419900]:
        #for step in [1900, 20900, 39900, 58900, 77900]:
        # for step in [1,3,5]:
        for step in STEPS:
            for light_direction in [0,1]:
                # output_dir= RENDER_DIR.format(DIR_NAME,version,step,light_direction)
                # input_dir = LIGHT_DIR.format(DIR_NAME,version,step,light_direction)
                # chk_dir = RENDER_DIR_NO_STEP.format(DIR_NAME,version,step)
                
                output_dir= RENDER_DIR.format(DIR_NAME,version,f"{step:06d}",light_direction)
                input_dir = LIGHT_DIR.format(DIR_NAME,version,f"{step:06d}",light_direction)
                chk_dir = RENDER_DIR_NO_STEP.format(DIR_NAME,version,f"{step:06d}")
                
                # output_dir= RENDER_DIR.format(version,f"{step:06d}",light_direction)
                # input_dir = LIGHT_DIR.format(version,f"{step:06d}",light_direction)
                # chk_dir = RENDER_DIR_NO_STEP.format(version,f"{step:06d}")

                counting = {
                    'left': 0,
                    'right': 0,
                    'left_ids': [],
                    'right_ids': [],
                }
                for filename in sorted(os.listdir(output_dir)):
                    if filename.endswith(".npy"):
                        light = np.load(os.path.join(output_dir, filename))
                        light = convert_to_grayscale(light.transpose())
                        if light[1] > 0:
                            counting['right'] += 1
                            counting['right_ids'].append(filename)
                        else:
                            counting['left'] += 1
                            counting['left_ids'].append(filename)
                out_path = f"{chk_dir}/counting_{light_direction}.json"
                print(out_path)
                with open(out_path, "w") as f:
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