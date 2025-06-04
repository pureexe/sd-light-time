#!/bin/bash
#SBATCH --error=output/logs/output/%j_1e-4_shadings150k_maxshading32_out.txt   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/%j_1e-4_shadings150k_maxshading32_err.txt  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=1e-4_shadings150k_maxshading32          #           V1_OLD: 111180 
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-4080
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON  112565 | 112552 / 112513


CONTAINER="/ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif"
LEARNING_RATE="1e-4"
RESOLUTION="512"
BATCH_SIZE="8"
GRAD_ACCUM="1"
CHECK_EVERY="5000"
NUM_EPOCH="100"
NUM_WORKER="8" #DATALOADER WORKER
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/pure/t1/checkpoints/sd-light-time/20250519_epoch_resample/controlnet/shadings150k_maxshading32/v1/${LEARNING_RATE}"
DATASET_PATH="/pure/t1/datasets/laion-shading/v4/huggingface/LaionShading150k"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --dataset_name=${DATASET_PATH} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --num_train_epochs ${NUM_EPOCH} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER} --max_shading_value 32"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
