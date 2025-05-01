#!/bin/bash
#SBATCH --error=output/logs/output/%j_out.txt   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/%j_err.txt  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=1e-4n32ctrl          # Job name: 1e-4 no clip face 111292 
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu



CONTAINER="/ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif"
LEARNING_RATE="1e-4"
RESOLUTION="512"
BATCH_SIZE="8"
GRAD_ACCUM="1"
CHECK_EVERY="20000"
NUM_EPOCH="100"
NUM_WORKER="8" #DATALOADER WORKER
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/20250425_huggingface_controlnet/controlnet/v1/${LEARNING_RATE}"
DATASET_PATH="/ist/ist-share/vision/pakkapon/relight/sd-light-time/output/datasets/laion-shading/v3/hf/LaionShading100k"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --dataset_name=${DATASET_PATH} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --num_train_epochs ${NUM_EPOCH} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER} --max_shading_value 32.0"

echo ${COMMAND}

singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
