#!/bin/bash
#SBATCH --error=output/logs/error/%j_1e-4max6_realgtalbedo_err.txt   # STDOUT error is written in slurm.out.JOBID
#SBATCH --output=output/logs/output/%j_1e-4max6_realgtalbedo_out.txt  # STDOUT output is written in slurm.err.JOBID
#SBATCH --job-name=1e-4max6realgtalbedo         #            
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON  

CONTAINER="/pure/f1/singularity/relight_20250510.sif"
LEARNING_RATE="1e-4"
RESOLUTION="512"
BATCH_SIZE="8"
GRAD_ACCUM="1"
CHECK_EVERY="3079"
NUM_STEPS="3079000" # 1000 EPOCH
NUM_WORKER="8" #DATALOADER WORKER
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/pure/t1/checkpoints/sd-light-time/20250605_controlnet_with_albedo/controlnet/multi_illumination_real_image_gt_shading_with_albedo/v0_hf_max6_44/${LEARNING_RATE}"
DATASET_PATH="/pure/f1/datasets/multi_illumination/real_image_gt_shading/v0/hf/MultiRealImageRealShadingAndAlbedo/MultiRealImageRealShadingAndAlbedo.py"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --dataset_name=${DATASET_PATH} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --max_train_steps ${NUM_STEPS} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER}  --resume_from_checkpoint latest --max_shading_value 6 --max_albedo_value 44 --num_controlnet_conditioning_channels 6"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist --nv --env HF_HUB_CACHE=/pure/t1/huggingface/hub/ --env HUB_HOME=/pure/t1/huggingface/ ${CONTAINER} ${COMMAND}
