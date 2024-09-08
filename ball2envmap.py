# render exr file to exposure of bracketed images

import time 
import argparse 
import os
from multiprocessing import Pool
from functools import partial
import numpy as np
import skimage
from tqdm.auto import tqdm
try:
    from hdrio import imread as exr_read
except:
    pass

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/data2/pakkapon/relight/sd-light-time/datasets/face/face2000_single/exr/01000" ,help='input_directory') 
    parser.add_argument("--output_dir", type=str, default="/data2/pakkapon/relight/sd-light-time/datasets/face/face2000_single/env_under/01000" ,help='directory to output environment map') #dataset name or directory 
    parser.add_argument("--gamma", type=float, default=2.4, help="gamma correction (Stylelight use 2.4) but 1.8 - 2.6 should be okay")
    parser.add_argument("--exposure_step", type=str, default="0, -5", help="Exposure Value (EV)")
    parser.add_argument("--threads", type=int, default=8, help="num thread for pararell processing")
    parser.add_argument('--auto_gamma', dest='auto_gamma', action='store_true')
    parser.set_defaults(auto_gamma=False)
    parser.add_argument('--save_alpha', dest='save_alpha', action='store_true')
    parser.set_defaults(save_alpha=False)
    return parser

def read_image(path):
    if path.endswith(".exr") or path.endswith(".hdr"):
        image = exr_read(path)
    elif path.endswith(".npy"):
        image = np.load(path)
        # flip from bgr to rgb
        image = image[...,::-1]
    else:
        raise ValueError("Unknown file type")
    return image

def apply_auto_gamma(image):
    arr = image.copy()
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return arr 

def undo_auto_gamma(image):
    arr = image.copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr

def process_image(args: argparse.Namespace, file_name: str):
    # read image
    image = read_image(os.path.join(args.input_dir, file_name))[...,:3]
    
    # get exposure step
    exposure_step = [int(x) for x in args.exposure_step.split(",")]
    
    # get image name
    image_name = file_name.split(".")[0]
    
    # get gamma value
    if args.auto_gamma:
        to_tonemap_image = apply_auto_gamma(image.copy())
    else:
        to_tonemap_image = image.copy()
    
    # get alpha
    hdr2ldr = TonemapHDR(gamma=args.gamma, percentile=99, max_mapping=0.9)
    ldr, alpha, _ = hdr2ldr(to_tonemap_image, gamma=not args.auto_gamma)

    
    if not args.auto_gamma and args.save_alpha:
        os.makedirs(f'{args.output_dir}/alpha', exist_ok=True)
        with open(f'{args.output_dir}/alpha/{image_name}.txt', 'w') as f:
            f.write(f"{alpha}\n")
    
    # save exposure
    for idx, exposure_value in enumerate(exposure_step):
        # apply exposure
        exposure_image = image * 2 ** exposure_value
        
        # apply gamma correction
        if args.auto_gamma:
            exposure_image = apply_auto_gamma(exposure_image)
        else:
            exposure_image = exposure_image ** (1/args.gamma)
        
        # apply alpha tone mapping (out = alpha*(in**(1/gamma)))
        exposure_image = alpha * exposure_image
                
        # clip to 0-1
        exposure_image = np.clip(exposure_image, 0, 1)
        
        # save image
        exposure_image = skimage.img_as_ubyte(exposure_image)
        if args.save_alpha:
            os.makedirs(args.output_dir+f"/ev{exposure_value}", exist_ok=True)
            output_path = os.path.join(args.output_dir+f"/ev{exposure_value}", f"{image_name}.png")
        else:
            output_path = os.path.join(args.output_dir, f"{image_name}_ev{exposure_value}.png")
        skimage.io.imsave(output_path, exposure_image, check_contrast=False)



def main():
    # running time measuring
    start_time = time.time()
    
    # load arguments
    args = create_argparser().parse_args()
    if args.auto_gamma:
        print("Using Skimage auto gamma")
    
    # make output directory if not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
     # get all file in the directory
    files = sorted(os.listdir(args.input_dir))
    
    # create partial function for pararell processing
    process_func = partial(process_image, args)
       
    # pararell processing
    with Pool(args.threads) as p:
        list(tqdm(p.imap(process_func, files), total=len(files)))
    
    # print total time 
    print("TOTAL TIME: ", time.time() - start_time)        
    

    
if __name__ == "__main__":
    main()