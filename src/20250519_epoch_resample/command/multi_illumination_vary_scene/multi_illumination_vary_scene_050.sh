#!/bin/bash
#SBATCH --error=output/logs/error/%j_vary050_err.txt   # STDOUT error is written in slurm.out.JOBID
#SBATCH --output=output/logs/output/%j_vary050_out.txt  # STDOUT output is written in slurm.err.JOBID
#SBATCH --job-name=50vary          #            
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON  115686

CONTAINER="/pure/f1/singularity/relight_20250510.sif"
LEARNING_RATE="1e-4"
RESOLUTION="512"
BATCH_SIZE="8"
GRAD_ACCUM="1"
CHECK_EVERY="3125"
NUM_STEPS="3125000" # 1000 EPOCH
NUM_WORKER="8" #DATALOADER WORKER
MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR="/pure/t1/checkpoints/sd-light-time/20250519_epoch_resample/controlnet/multi_illuination_vary_scene/v1/num_scene_050"
DATASET_PATH="/pure/f1/datasets/multi_illumination/lstsq_image_lstsq_shading/v0/hf/MultiLstShadingScene050/MultiLstShadingScene050.py"
COMMAND="accelerate launch train_controlnet.py --pretrained_model_name_or_path=${MODEL_DIR} --output_dir=${OUTPUT_DIR} --dataset_name=${DATASET_PATH} --resolution=${RESOLUTION} --learning_rate=${LEARNING_RATE} --train_batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --checkpointing_steps ${CHECK_EVERY} --max_train_steps ${NUM_STEPS} --use_8bit_adam --mixed_precision fp16 --dataloader_num_workers ${NUM_WORKER}  --resume_from_checkpoint latest"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
