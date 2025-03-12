#!/bin/bash
#SBATCH --error=output/logs/output/%j   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/%j  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=face_light          # Job name
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-4080
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu


export SINGULARITYENV_PYTHONPATH=$(pwd):$(pwd)/src 
export SINGULARITYENV_HF_HOME=/ist/users/$USER/.cache/huggingface
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_HF_HUB_HOME=/ist/ist-share/vision/huggingface/
export SINGULARITYENV_HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/

CONTAINER="/pure/t1/singularity/diffusers0310v6.sif"
COMMAND="python src/20250221_optmized_shading_exr/train.py -lr 1e-4 --guidance_scale 1.0 --network_type sd_no_bg --batch_size 8 -c 1 --feature_type clip --shadings_dir \"shadings\" --backgrounds_dir \"backgrounds\" --images_dir \"images\" -dataset_val \"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial\" -dataset_val_split \"/data/pakkapon/datasets/face/ffhq_defareli/valid_spatial/index-array.json\""

singularity exec --bind /pure/t1/datasets:/data/pakkapon/datasets --bind /ist:/ist --nv ${CONTAINER} ${COMMAND}
