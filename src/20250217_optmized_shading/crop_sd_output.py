import skimage
from tqdm.auto import tqdm
import os

ROOT_DIR = os.path.abspath("/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250120_efficient_shading/val_all_everett_dining1_blender_orthographic_shading/default/1.0/blender_orthographic/1e-4")

def get_version(logdir):
    versions = os.listdir(logdir)
    return versions[0]

def main():
    for chk in tqdm(range(1,51)):
        logdir = f"{ROOT_DIR}/chk{chk}/lightning_logs"
        version = get_version(logdir)
        gt_dir = f"{logdir}/{version}/with_groudtruth/"
        sd_dir = f"{logdir}/{version}/sd_output/"
        os.makedirs(sd_dir, exist_ok=True)
        files = sorted(os.listdir(gt_dir))
        for f in files:
            img = skimage.io.imread(f"{gt_dir}/{f}")
            img = img[516:1028, 2:516]
            skimage.io.imsave(f"{sd_dir}/{f}", img)

if __name__ == "__main__":
    main()  