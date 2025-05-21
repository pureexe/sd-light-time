import os 

SCENES = ["everett_kitchen6","everett_dining1","everett_kitchen2","everett_kitchen4"]
CHROMEBALL_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/chromeball/"
GRPOUND_TRUTH_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/control_render_from_fitting_v2/"
SHADING_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/control_shading_from_fitting_v3_norm/"
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/video"

def main():
    for scene in SCENES:
        chromeball_path = CHROMEBALL_DIR + f"/{scene}/%02d.png" 
        ground_truth_path = GRPOUND_TRUTH_DIR + f"/{scene}/dir_%d_mip2.png" 
        shading_path = SHADING_DIR + f"/{scene}/dir_%d_mip2.png"
        
        if not os.path.exists(CHROMEBALL_DIR + f"/{scene}") or not os.path.exists(GRPOUND_TRUTH_DIR + f"/{scene}"):
            print(f"Missing files for scene: {scene}")
            continue
        
        video_dir = os.path.join(OUTPUT_DIR, scene)
        os.makedirs(video_dir, exist_ok=True)
        os.chmod(video_dir, 0o777)
        
        # # Create video with chromeball
        os.system(f"ffmpeg -framerate 10 -i {chromeball_path} -c:v libx264 -pix_fmt yuv420p {video_dir}/{scene}_chromeball.mp4")
        
        # Create video with ground truth
        os.system(f"ffmpeg -framerate 10 -i {ground_truth_path} -c:v libx264 -pix_fmt yuv420p {video_dir}/{scene}_ground_truth.mp4")

        # Create video with ground truth
        os.system(f"ffmpeg -framerate 10 -i {shading_path} -c:v libx264 -pix_fmt yuv420p {video_dir}/{scene}_shading.mp4")


if __name__ == "__main__":
    main()