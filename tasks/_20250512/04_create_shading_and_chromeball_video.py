import subprocess

SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]

def main():
    for scene in SCENES:
        # create shading video 
        subprocess.run([f'ffmpeg -r 10 -i shading_exr_perspective_v3_order6_viz_max/dir_%d_mip2.png -c:v libx264 -crf 12 -pix_fmt yuv420p shadings.mp4'], cwd=f'/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}', shell=True)        
        # create chromeball video
        subprocess.run([f'ffmpeg -r 10 -i shading_exr_perspective_v3_order6_ball_viz_max/dir_%d_mip2.png -c:v libx264 -crf 12 -pix_fmt yuv420p chromeball.mp4'], cwd=f'/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output_t1/multi_illumination/real/rotate/{scene}', shell=True)

    
if __name__ == "__main__":
    main()