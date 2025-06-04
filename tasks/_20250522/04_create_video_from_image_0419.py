import os 
SCENES = ["everett_kitchen6","everett_dining1","everett_kitchen2","everett_kitchen4"]
# VERSION  = ["115598","115595","115596","115597"] # CHK59
# CHECKPOINTS = [59]
VERSION  = ["115598","115599","115600","115601"] # CHK59
CHECKPOINTS = [39]
for scene_id, scene in enumerate(SCENES):
    version = VERSION[scene_id]
    for checkpoint in CHECKPOINTS:
        predict_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250419_train_in_the_wild/val_rotate_{scene}/default/1.0/no_clip/1e-4/chk{checkpoint}/lightning_logs/version_{version}"
        print(predict_path)
        video_dir = os.path.join(f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250419_train_in_the_wild/val_rotate_everett_dining1', "video")
        video_path = f"{video_dir}/with_groundtruth/{scene}_{checkpoint}.mp4"
        if not os.path.exists(predict_path):
            continue
        if os.path.exists(video_path):
            print("EXISTING:", video_path)
            continue
        count_image = len(os.listdir(f"{predict_path}/with_groudtruth/"))
        if count_image < 60:
            continue
        os.makedirs(video_dir, exist_ok=True) 
        os.chmod(video_dir, 0o777)  
        # compute with ground truth dir
        withgt_dir = os.path.join(video_dir, "with_groundtruth")  
        os.makedirs(withgt_dir, exist_ok=True)
        os.chmod(withgt_dir, 0o777)
        os.system(f"ffmpeg -framerate 10 -i {predict_path}/with_groudtruth/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.jpg -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{checkpoint}.mp4")
        # compute with predict dir
        withgt_dir = os.path.join(video_dir, "predict")     
        os.makedirs(withgt_dir, exist_ok=True)
        os.chmod(withgt_dir, 0o777)
        os.system(f"ffmpeg -framerate 10 -i {predict_path}/crop_image/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.png -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{checkpoint}.mp4")
