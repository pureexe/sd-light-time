import os

# List of input video filenames
inputs = [
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/oldcode_chk40_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run1_chk40_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run2_chk40_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run3_chk40_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen6_least_square_shading/video/predict/everett_kitchen6_least_square_shading_1e-4_lstsq_image_lstsq_shading_40.mp4", # new code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/oldcode_chk60_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run1_chk60_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run2_chk60_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run3_chk60_seed42.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen6_least_square_shading/video/predict/everett_kitchen6_least_square_shading_1e-4_lstsq_image_lstsq_shading_60.mp4", # new code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/ground_truth.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/chromeball.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/shading.mp4"
]

# Create input flags for FFmpeg
input_flags = " ".join(f"-i {name}" for name in inputs)

# Define the FFmpeg filter complex string
filter_complex = """
nullsrc=size=1280x768 [base];
[0:v] scale=256:256 [v0];  [1:v] scale=256:256 [v1];  [2:v] scale=256:256 [v2];
[3:v] scale=256:256 [v3];  [4:v] scale=256:256 [v4];  [5:v] scale=256:256 [v5];
[6:v] scale=256:256 [v6];  [7:v] scale=256:256 [v7];  [8:v] scale=256:256 [v8];
[9:v] scale=256:256 [v9];  [10:v] scale=256:256 [v10]; [11:v] scale=256:256 [v11];
[12:v] scale=256:256 [v12];
color=c=black:size=256x256:d=1:r=30 [black1];
color=c=black:size=256x256:d=1:r=30 [black2];
[v0][v1][v2][v3][v4][v5][v6][v7][v8][v9][v10][v11][v12][black1][black2] \
xstack=inputs=15:layout=0_0|256_0|512_0|768_0|1024_0|\
0_256|256_256|512_256|768_256|1024_256|\
0_512|256_512|512_512|768_512|1024_512:fill=black[out]
"""

# Clean up line breaks and spaces
filter_complex = " ".join(line.strip() for line in filter_complex.strip().splitlines())

# Final FFmpeg command
cmd = f"ffmpeg {input_flags} -filter_complex \"{filter_complex}\" -map \"[out]\" -c:v libx264 -preset fast -crf 23 -y output/kitchen6_40_60_oldcode_r1r2r3_webdataset.mp4"

# Run the command
os.system(cmd)