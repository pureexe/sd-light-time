import torch
import skimage
import os 
from tqdm.auto import tqdm
from create_chromeball_to_visualize import ChromeballRenderer

def read_image(image_path):
    image = skimage.io.imread(image_path)
    image = skimage.img_as_float(image)
    image = torch.tensor(image).permute(2,0,1).to('cuda')
    return image


@torch.inference_mode()
def main():
    IMAGE_DIR = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/images"
    SH_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/shcoeff_order2_from_fitting/{}.npy"
    IMAGE_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/images/{}/dir_0_mip2.jpg"
    ALBEDO_PATH = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/test/control_albedo_from_fitting_v2/{}.png"
    
    OUTPUT_SCENE = "/ist/ist-share/vision/relight/datasets/multi_illumination/spherical/val_rotate_test_scenes"
    
    SHADING_DIR = f"{OUTPUT_SCENE}/control_shading_from_fitting_v3_exr"
    SHADING_NORM_DIR = f"{OUTPUT_SCENE}/control_shading_from_fitting_v3_norm"
    RENDER_DIR = f"{OUTPUT_SCENE}/control_render_from_fitting_v2"
    CHROMEBALL_DIR = f"{OUTPUT_SCENE}/chromeball"
    LIGHT_ID = 22

    #scenes = os.listdir(IMAGE_DIR)
    scenes = ['everett_kitchen4']
    renderer = ChromeballRenderer()
    for scene in tqdm(scenes):
        sh_path = SH_PATH.format(scene)
        renderer.load_shcoeffs(sh_path)
        renderer.render_rotated_chromeball(
            chromeball_dir = os.path.join(CHROMEBALL_DIR, scene),
            light_id=LIGHT_ID
        )

        # image_path
        image_path = IMAGE_PATH.format(scene) 
        image = read_image(image_path)

        renderer.normal_map = None 
        renderer.get_normal(image)
        
        # albedo path 
        albedo_path = ALBEDO_PATH.format(scene)
        albedo = read_image(albedo_path)
        renderer.set_albedo(albedo)
        
        renderer.render_scene(
            shading_dir=os.path.join(SHADING_DIR, scene),
            shading_norm_dir=os.path.join(SHADING_NORM_DIR, scene),
            render_dir=os.path.join(RENDER_DIR, scene),
            light_id=LIGHT_ID
        )
    
    
if __name__ == "__main__":
    main()