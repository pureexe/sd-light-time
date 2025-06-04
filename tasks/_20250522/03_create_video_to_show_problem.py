import subprocess

# Generate input file names
# KITCHEN 6
# input_files = [
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/oldcode_chk40_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run1_chk40_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run2_chk40_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run3_chk40_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen6_least_square_shading/video/predict/everett_kitchen6_least_square_shading_1e-4_lstsq_image_lstsq_shading_40.mp4", # new code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/oldcode_chk60_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run1_chk60_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run2_chk60_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/rotate_run3_chk60_seed42.mp4", #old code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen6_least_square_shading/video/predict/everett_kitchen6_least_square_shading_1e-4_lstsq_image_lstsq_shading_60.mp4", # new code chk-40
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/ground_truth.mp4",
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/chromeball.mp4",
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/shading.mp4",
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/black_video.mp4",
#     "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/black_video.mp4"
# ]
# KITCHEN 2
input_files = [
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run1_40.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run1_40.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run2_40.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run3_40.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen2_least_square_shading/video/predict/everett_kitchen2_least_square_shading_1e-4_lstsq_image_lstsq_shading_40.mp4", # new code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run1_60.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run1_60.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run2_60.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/predict/everett_kitchen2_seed_rotate_run3_60.mp4", #old code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output_t1/20250519_epoch_resample/val_rotate_everett_kitchen2_least_square_shading/video/predict/everett_kitchen2_least_square_shading_1e-4_lstsq_image_lstsq_shading_50.mp4", # new code chk-40
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/everett_kitchen2_ground_truth.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/everett_kitchen2_chromeball.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen2/video/everett_kitchen2_shading.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/black_video.mp4",
    "/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/val_rotate_everett_kitchen6/video_differnet_train/black_video.mp4"
]

# Target resolution (adjust to your needs)
target_width = 512
target_height = 512
target_fps = 10

# Prepare input arguments
input_args = []
for file in input_files:
    input_args.extend(["-i", file])

# Build filter chains for each input
filter_chains = []
for i in range(15):
    chain = (
        f"[{i}:v]scale={target_width}:{target_height},fps={target_fps},format=yuv420p[v{i}]"
    )
    filter_chains.append(chain)

# Layout for 5 columns and 3 rows
layout_parts = []
for row in range(3):
    for col in range(5):
        layout_parts.append(f"{col * target_width}_{row * target_height}")
layout = '|'.join(layout_parts)

# Combine into one filter_complex string
filter_complex = ';'.join(filter_chains)
filter_complex += f";{''.join(f'[v{i}]' for i in range(15))}xstack=inputs=15:layout={layout}[out]"

# Build full command
cmd = ["ffmpeg"] + input_args + [
    "-filter_complex", filter_complex,
    "-map", "[out]",
    "-vsync", "vfr",
    "-preset", "fast",
    "-crf", "18",
    "output_kitchen2.mp4"
]

# Execute the command
subprocess.run(cmd)
