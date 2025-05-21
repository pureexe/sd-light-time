import os 

SCENES = ["everett_kitchen6","everett_dining1","everett_kitchen2","everett_kitchen4"]
OUTPUT_DIR = "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/video_compare_chk80"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for scene in SCENES:
        chromeball_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/video/{scene}/{scene}_chromeball.mp4"
        ground_truth_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/video/{scene}/{scene}_ground_truth.mp4"
        shading_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/multi_illumination/spherical/val_rotate_test_scenes/video/{scene}/{scene}_shading.mp4"
        code0425_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_{scene}/video/predict/{scene}_batch_8_v1_r1_80.mp4"
        code0520_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_{scene}/video/predict/{scene}_code0510_least_square_1e-4_80.mp4"
        
        # Create video with chromeball
        cmd = f'ffmpeg -i {ground_truth_path} -i {code0425_path} -i {code0520_path} -i {chromeball_path} -i {shading_path} -f lavfi -t 1 -i color=black:s=512x512:r=10 -filter_complex "xstack=inputs=6:layout=0_0|512_0|1024_0|0_512|512_512|1024_512[v]" -map "[v]" -c:v libx264 -crf 23 -preset veryfast {OUTPUT_DIR}/{scene}.mp4'
        os.system(cmd)
        
if __name__ == "__main__":
    main()