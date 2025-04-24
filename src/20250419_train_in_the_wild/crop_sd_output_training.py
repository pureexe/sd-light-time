import skimage
from tqdm.auto import tqdm
import os

ROOT_DIR = os.path.abspath("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250221_optmized_shading_exr/multi_mlp_fit/lightning_logs")
TARGET_VERSION = ["106222","106223","106224","106225","106259","106260","106261","106262"]
#ROOT_DIR = os.path.abspath("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250419_train_in_the_wild/multi_mlp_fit/lightning_logs")
#TARGET_VERSION = ["106314","106315","106625","106626","106627","106628","106629","106630","106631","106632"]
#TARGET_VERSION = ["106625","106626","106627","106628","106629","106630","106631","106632"]
def get_version(logdir):
    versions = os.listdir(logdir)
    return versions[0]

def main():
    for version in TARGET_VERSION:
        epoch_dir = [a for a in sorted(os.listdir(f"{ROOT_DIR}/version_{version}")) if a.startswith("step")]
        print(f"{ROOT_DIR}/version_{version}")
        for epoch in tqdm(epoch_dir):
            gt_dir = f"{ROOT_DIR}/version_{version}/{epoch}/with_groudtruth/"
            sd_dir = f"{ROOT_DIR}/version_{version}/{epoch}/sd_output/"
            os.makedirs(sd_dir, exist_ok=True)
            os.chmod(sd_dir, 0o777)
            files = sorted(os.listdir(gt_dir))
            for f in files:
                out_path = f"{sd_dir}/{f}"
                if os.path.exists(out_path):
                    continue
                img = skimage.io.imread(f"{gt_dir}/{f}")
                img = img[516:1028, 2:516]
                
                skimage.io.imsave(out_path, img)
                os.chmod(out_path, 0o777)

if __name__ == "__main__":
    main()  