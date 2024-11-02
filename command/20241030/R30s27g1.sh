#!/bin/bash
#SBATCH --error=output/logs/error/error.%j   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/error.%j  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=R30s27g1          # Job name
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu


export SINGULARITYENV_PYTHONPATH=$(pwd):$(pwd)/src 
export SINGULARITYENV_HF_HOME=/ist/users/$USER/.cache/huggingface
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_HF_HUB_HOME=/ist/ist-share/vision/huggingface/
export SINGULARITYENV_HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/

CONTAINER="/ist/ist-share/vision/pakkapon/singularity/lighttime-deepfloyd.sif"
COMMAND="python src/20241027/val_coeff27.py -m faceval10k_fuse_test_right -c 30 -g 1.0"

singularity exec --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --bind /ist:/ist --nv ${CONTAINER} ${COMMAND}
