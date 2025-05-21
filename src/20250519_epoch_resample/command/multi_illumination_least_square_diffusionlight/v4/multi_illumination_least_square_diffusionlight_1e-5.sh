#!/bin/bash
#SBATCH --error=output/logs/error/%j_err.txt   # STDOUT error is written in slurm.out.JOBID
#SBATCH --output=output/logs/output/%j_out.txt  # STDOUT output is written in slurm.err.JOBID
#SBATCH --job-name=v41e-5LeastSQMulti          #            
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-4080
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON   6827

CONTAINER="/pure/f1/singularity/relight_20250510.sif"
LEARNING_RATE="1e-5"
RESOLUTION="512"
BATCH_SIZE="8"
GRAD_ACCUM="1"
CHECK_EVERY="3079"
NUM_STEPS="3079000" # 1000 EPOCH
NUM_WORKER="8" #DATALOADER WORKER
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/pure/t1/checkpoints/sd-light-time/20250519_epoch_resample/controlnet/multi_illumination_least_square_diffusionlight/v4/${LEARNING_RATE}"
TRAIN_DATA_DIR="/pure/f1/datasets/multi_illumination/least_square/v4_webdataset/train/train-{0000..024}.tar"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --train_data_dir=${TRAIN_DATA_DIR} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --max_train_steps ${NUM_STEPS} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER}  --resume_from_checkpoint latest --estimated_dataset_length 24625"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
