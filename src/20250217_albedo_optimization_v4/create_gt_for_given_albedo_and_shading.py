import skimage 
import numpy as np 
import ezexr 
from tqdm.auto import tqdm 
import os 

def main():
    albedo_path = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/train/control_albedo_from_fitting_v2/14n_copyroom10.png"
    shading_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_copyroom10/control_shading_from_fitting_v3_exr/14n_copyroom10"
    output_dir = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_copyroom10/control_render_from_fitting_v2/14n_copyroom10"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(shading_dir))

    # read albedo 
    albedo = skimage.io.imread(albedo_path)
    albedo = skimage.img_as_float(albedo) # re-range to [0,1]

    for filename in tqdm(files):
        # read shading 
        shading = ezexr.imread(os.path.join(shading_dir, filename))
        out_img = albedo[...,:3] * shading[...,:3] 
        # cap-to 0,1
        out_img = np.clip(out_img, 0, 1)
        out_img = skimage.img_as_ubyte(out_img)
        skimage.io.imsave(os.path.join(output_dir, filename.replace('.exr','.jpg')), out_img)
    
    print("DONE SAVE IMAGE")
        
    
if __name__ == "__main__":
    main()