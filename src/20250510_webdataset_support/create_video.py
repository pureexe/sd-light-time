import os 

def main():
    for seed in ['42']:
        named_seed = '100'
        for run_id, dir_name in enumerate(['rotate_run1', 'rotate_run2', 'rotate_run3', 'rotate_run4', 'rotate_run5']):
            for ckpt in ['35', '40', '60', '80']:
                in_dir = f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/{dir_name}/run_{run_id+1:02d}_chk{ckpt}/seed{named_seed}/crop_image/'
                out_file = f'/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/{dir_name}_chk{ckpt}_seed{named_seed}.mp4'
                if os.path.exists(out_file):
                    continue
                if not os.path.exists(in_dir):
                    continue
                os.system(f'ffmpeg -r 10 -i {in_dir}/everett_kitchen6-dir_0_mip2_everett_kitchen6-dir_%d_mip2.png -c:v libx264 -crf 12 -pix_fmt yuv420p {out_file}')

if __name__ == "__main__":
    main()