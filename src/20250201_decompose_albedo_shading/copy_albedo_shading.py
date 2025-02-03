import argparse 
import os
from tqdm.auto import tqdm 
import shutil 

parser = argparse.ArgumentParser(description="Process index and total.")
parser.add_argument('-i', '--index', type=int, default=0, help='Index of the item')
parser.add_argument('-t', '--total', type=int, default=1, help='Total number of items')

args = parser.parse_args()


def main():
    root_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test"
    image_dir = "images"
    albedo_dir = "control_intrinsic_albedo"
    albedo_shared_dir = "control_intrinsic_albedo_shared"
    mode = 'bae'
    ORDER = 2

 
    scenes = sorted(os.listdir(os.path.join(root_dir, image_dir)))

    output_albedo_shared_dir = os.path.join(root_dir,albedo_shared_dir)
    os.makedirs(output_albedo_shared_dir, exist_ok=True)
    os.chmod(output_albedo_shared_dir, 0o777)
 
    print("CREATING QUEUES...")
    queues  = []
    for scene in scenes:
        for idx in range(25):
            queues.append((scene,idx))

    pbar = tqdm(queues[args.index::args.total])
    pbar.set_description(f"")

    for info in pbar:
        
        pbar.set_postfix(item=f"{info[0]}/{info[1]}")

        idx = info[1]
        scene = info[0]
        output_albedo_dir = os.path.join(root_dir,albedo_dir,scene)
        output_albedo_shared_dir = os.path.join(root_dir,albedo_shared_dir,scene)
        os.makedirs(output_albedo_shared_dir, exist_ok=True)
        os.chmod(output_albedo_shared_dir, 0o777)
        albedo_shared_path = os.path.join(output_albedo_shared_dir, f"dir_{idx}_mip2.png")
        if os.path.exists(albedo_shared_path):
            print("SKIPPING")
            continue

        albedo_source_path = os.path.join(output_albedo_dir, f"dir_0_mip2.png")
        shutil.copy2(albedo_source_path, albedo_shared_path)
        os.chmod(albedo_shared_path, 0o777)



if __name__ == "__main__":
    main()