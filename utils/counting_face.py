import os
import numpy as np
import json 
STEPS = [22000, 42000]

#DIR_NAME =  "20240604_TimeEmbedingV2"
DIR_NAME = "20240612/multi_fit_manual"
LIGHT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face/step{}/{}"
RENDER_DIR ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face_light/step{}/{}"
RENDER_DIR_NO_STEP ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face_light/step{}"

# DIR_NAME = "20240609/multi_fit_val/1e-3_ep41"
# LIGHT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face/step000000/guidance_{}/{}"
# RENDER_DIR ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face_light/step000000/guidance_{}/{}"
# RENDER_DIR_NO_STEP ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/output/{}/lightning_logs/version_{}/face_light/step000000/guidance_{}"

# DIR_NAME =  "20240604_TimeEmbedding"
# LIGHT_DIR = "/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/lightning_logs/version_{}/face/step{}/{}"
# RENDER_DIR ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/lightning_logs/version_{}/face_light/step{}/{}"
# RENDER_DIR_NO_STEP ="/home/pakkapon/mnt_tl_vision17/data2/pakkapon/relight/sd-light-time/lightning_logs/version_{}/face_light/step{}"


def main():
    # for version in range(4,8):
    #     for step in [77900]:
    #         for light_direction in [0,1]:
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