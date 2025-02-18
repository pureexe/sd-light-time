import skimage
from tqdm.auto import tqdm
import os

ROOT_DIR = os.path.abspath("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/multi_mlp_fit/lightning_logs")
TARGET_VERSION = ["98472"]
def get_version(logdir):
    versions = os.listdir(logdir)
    return versions[0]

def main():
    for version in TARGET_VERSION:
        epoch_dir = [a for a in sorted(os.listdir(f"{ROOT_DIR}/version_{version}")) if a.startswith("epoch")]
        for epoch in tqdm(epoch_dir):
            gt_dir = f"{ROOT_DIR}/version_{version}/{epoch}/with_groudtruth/"
            sd_dir = f"{ROOT_DIR}/version_{version}/{epoch}/sd_output/"
            os.makedirs(sd_dir, exist_ok=True)
            files = sorted(os.listdir(gt_dir))
            for f in files:
                img = skimage.io.imread(f"{gt_dir}/{f}")
                img = img[516:1028, 2:516]
                skimage.io.imsave(f"{sd_dir}/{f}", img)


if __name__ == "__main__":
    main()  