import os 
SCENES = ["everett_dining1","everett_kitchen2","everett_kitchen4","everett_kitchen6"]
EXPERIMENT = ['code0510_least_square_1e-4', 'code0510_least_square_1e-5']
CHECKPOINTS = range(40, 70, 10)

for scene in SCENES:
    for experiment in EXPERIMENT:
        for checkpoint in CHECKPOINTS:
            predict_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_{scene}/{experiment}/chk{checkpoint}/seed42"
            if not os.path.exists(predict_path):
                continue
            video_dir = f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/video'
            os.makedirs(video_dir, exist_ok=True) 
            os.chmod(video_dir, 0o777)  
            # compute with ground truth dir
            withgt_dir = os.path.join(video_dir, "with_groundtruth")     
            os.makedirs(withgt_dir, exist_ok=True)
            os.chmod(withgt_dir, 0o777)
            os.system(f"ffmpeg -framerate 10 -i {predict_path}/with_groudtruth/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.jpg -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{experiment}_{checkpoint}.mp4")
            # compute with predict dir
            withgt_dir = os.path.join(video_dir, "predict")     
            os.makedirs(withgt_dir, exist_ok=True)
            os.chmod(withgt_dir, 0o777)
            os.system(f"ffmpeg -framerate 10 -i {predict_path}/crop_image/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.png -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{experiment}_{checkpoint}.mp4")
        