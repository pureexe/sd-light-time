import os 
SCENES = ["everett_kitchen6","everett_dining1","everett_kitchen2","everett_kitchen4"]
EXPERIMENT = ['1e-4_lstsq_image_lstsq_shading','1e-4_real_image_lstsq_shading','1e-5_lstsq_image_lstsq_shading','1e-5_real_image_lstsq_shading']
SHADINGS = ["diffusionlight_shading","least_square_shading"]
CHECKPOINTS = [60]
for scene in SCENES:
    for shading in SHADINGS:  
        for experiment in EXPERIMENT:
            for checkpoint in CHECKPOINTS:
                predict_path = f"/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_{scene}_{shading}/{experiment}/chk{checkpoint}/seed42"
                print(predict_path)
                video_dir = os.path.join(f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_{scene}_{shading}', "video")
                video_path = f"{video_dir}/with_groundtruth/{scene}_{shading}_{experiment}_{checkpoint}.mp4"
                if not os.path.exists(predict_path):
                    continue
                if os.path.exists(video_path):
                    print("EXISTING")
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
                os.system(f"ffmpeg -framerate 10 -i {predict_path}/with_groudtruth/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.jpg -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{shading}_{experiment}_{checkpoint}.mp4")
                # compute with predict dir
                withgt_dir = os.path.join(video_dir, "predict")     
                os.makedirs(withgt_dir, exist_ok=True)
                os.chmod(withgt_dir, 0o777)
                os.system(f"ffmpeg -framerate 10 -i {predict_path}/crop_image/{scene}-dir_0_mip2_{scene}-dir_%d_mip2.png -c:v libx264 -pix_fmt yuv420p {withgt_dir}/{scene}_{shading}_{experiment}_{checkpoint}.mp4")
        