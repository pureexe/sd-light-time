#!/bin/bash
#SBATCH --error=output/logs/error/render_%j_err.txt   # STDOUT error is written in slurm.out.JOBID
#SBATCH --output=output/logs/output/render_%j_out.txt  # STDOUT output is written in slurm.err.JOBID
#SBATCH --job-name=t1e-4k6           
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu
#SBATCH --cpus-per-task=8            # The number of CPU cores
# RUN ON   113149


CONTAINER="/pure/f1/singularity/relight_20250510.sif"
COMMAND="python inference.py -p 1e-4 -m everett_kitchen6"

echo ${COMMAND}

singularity exec --bind /pure:/pure --bind /ist:/ist --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
