import os
import numpy as np
import json 


DATASET_NAME = "val_coeff27_faceval10k_fuse_test_right"
GUIDANCE_SCALES = ['1.0', '3.0', '5.0', '7.0']
LEARNING_RATES = ['0.0001']
CHECKPOINTS = [0, 10, 20]

#ROOT_DIR = "../../output/20241027/{}/{}/no_control/{}/chk{}/lightning_logs"
ROOT_DIR = "../../output/20241027/{}/no_control/{}/no_control/{}/chk{}/lightning_logs"

def main():

    for learning_rate in LEARNING_RATES:
        for checkpoint in CHECKPOINTS:
            for guidance_scale in GUIDANCE_SCALES:
                root_dir = ROOT_DIR.format(DATASET_NAME, guidance_scale, learning_rate, checkpoint)
                print(root_dir)
                version_dir = os.listdir(root_dir)[0]
                face_dir = root_dir + '/' + version_dir+ "/face_light"
                counting_file = f"{root_dir}/counting.json"
                counting = {
                    'left': 0,
                    'right': 0,
                    'left_ids': [],
                    'right_ids': [],
                }
                found_the_file = 0
                if not os.path.exists(face_dir):
                    print(f"Cannot find {face_dir}")
                    continue
                for filename in sorted(os.listdir(face_dir)):
                    if filename.endswith(".npy"):
                        found_the_file += 1
                        light = np.load(os.path.join(face_dir, filename))
                        light = convert_to_grayscale(light.transpose())
                        if light[1] > 0:
                            counting['right'] += 1
                            counting['right_ids'].append(filename)
                        else:
                            counting['left'] += 1
                            counting['left_ids'].append(filename)
                if found_the_file < 100:
                    print(f"Found only {found_the_file} files in {face_dir}")
                    continue
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