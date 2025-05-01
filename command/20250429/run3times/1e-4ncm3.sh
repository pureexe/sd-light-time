#!/bin/bash
#SBATCH --error=output/logs/output/%j_out.txt   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/%j_err.txt  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=1e-4ncm3          # Job name: 1e-4 no clip multi-ilumn #111018  
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu



CONTAINER="/ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif"
LEARNING_RATE="1e-4"
NETWORK_TYPE="sd_only_shading"
COMMAND="python src/20250419_train_in_the_wild/train.py -lr ${LEARNING_RATE} --guidance_scale 1.0 --network_type ${NETWORK_TYPE} --batch_size 8 -c 5 --feature_type clip --shadings_dir control_shading_from_fitting_v3_exr --backgrounds_dir control_render_from_fitting_v2 --images_dir control_render_from_fitting_v2 -dataset /data/pakkapon/datasets/multi_illumination/spherical/train  -dataset_val /data/pakkapon/datasets/multi_illumination/spherical/val -dataset_val_split /data/pakkapon/datasets/multi_illumination/spherical/split-val-relight-light-array.json"

echo ${COMMAND}

singularity exec --bind /ist:/ist --bind /pure:/pure --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
