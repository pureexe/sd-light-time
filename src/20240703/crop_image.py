import os 
from tqdm.auto import tqdm
import skimage
from multiprocessing import Pool

#CHK_PTS = [19, 39, 59, 79, 99, 119]
CHK_PTS = [139, 159, 179, 199, 219, 239, 259]
LRS = ['5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5', '5e-3', '5e-4', '5e-5']
NAMES = ['dinov2', 'dinov2', 'dinov2', 'vae', 'vae', 'vae', 'slimnet', 'slimnet', 'slimnet']
VAL_FILES = ['light_x_minus', 'light_x_plus', 'light_y_minus', 'light_y_plus', 'light_z_minus', 'light_z_plus']

def process_file(meta):
    # create output dir 
    os.makedirs(meta["out_dir"], exist_ok=True)
    img = skimage.io.imread(meta['in_path'])
    img = img[2:512+2,516:516+512]
    skimage.io.imsave(meta['out_path'], img)

def main():
    process_files = []
    for version in range(9):
        lr = LRS[version]
        name = NAMES[version]
        for chk_pt in CHK_PTS:
            #print(f"Processing {name} {lr} {chk_pt}")
            for val_dir in VAL_FILES:
                dir_path = f"output/20240703/val_axis/{name}/{lr}/chk{chk_pt}/{val_dir}/lightning_logs/version_0/"
                in_dir = os.path.join(dir_path, "rendered_image")
                out_dir = os.path.join(dir_path, "crop_image")
                files = sorted(os.listdir(in_dir))
                for filename in files:
                    process_files.append({
                        "in_dir": in_dir,
                        "out_dir": out_dir,
                        "in_path": os.path.join(in_dir, filename),
                        "out_path": os.path.join(out_dir, filename)
                    })
    with Pool(32) as p:
      r = list(tqdm(p.imap(process_file, process_files), total=len(process_files)))



if __name__ == "__main__":
    main()