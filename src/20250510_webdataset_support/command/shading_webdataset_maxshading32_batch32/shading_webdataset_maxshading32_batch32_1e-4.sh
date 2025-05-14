#!/bin/bash
#SBATCH --error=output/logs/error/%j_err.txt   # STDOUT error is written in slurm.out.JOBID
#SBATCH --output=output/logs/output/%j_out.txt  # STDOUT output is written in slurm.err.JOBID
#SBATCH --job-name=1e-4n32web4_shading          #           
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON   114927/113149


CONTAINER="/pure/f1/singularity/relight_20250510.sif"
LEARNING_RATE="1e-4"
RESOLUTION="512"
BATCH_SIZE="1"
GRAD_ACCUM="32"
TOTAL_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
CHECK_EVERY="1000"
NUM_STEPS="150000000" # 100 EPOCH
NUM_WORKER="8" #DATALOADER WORKER
MAX_SHADING="32"
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/pure/t1/checkpoints/sd-light-time/20250510_webdataset_support/controlnet/shading_webdataset_maxshading${MAX_SHADING}_batch${TOTAL_BATCH}/v1/${LEARNING_RATE}"
DATASET_PATH="/pure/t1/datasets/laion-shading/v4/huggingface/LaionShading150k"
TRAIN_DATA_DIR="/pure/f1/datasets/laion-shading/v4_webdataset/train/train-{0000..0149}.tar"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --train_data_dir=${TRAIN_DATA_DIR} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --max_train_steps ${NUM_STEPS} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER} --max_shading_value=${MAX_SHADING}"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
