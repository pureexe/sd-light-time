#!/bin/bash
#SBATCH --error=output/logs/output/%j_out.txt   # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=output/logs/error/%j_err.txt  # STDOUT error is written in slurm.err.JOBID
#SBATCH --job-name=5e-4w3prompt          # Job name
#SBATCH --mem=64GB                   # Memory request for this job
#SBATCH --nodes=1                    # The number of nodes
#SBATCH --partition=gpu-cluster
#SBATCH --account=vision
#SBATCH --time=72:0:0                 # Running time 2 hours
#SBATCH --gpus=1                     # The number of gpu



CONTAINER="/ist/ist-share/vision/pakkapon/singularity/diffusers0310v6.sif"
LEARNING_RATE="5e-4"
COMMAND="python src/20250221_optmized_shading_exr/train.py -lr ${LEARNING_RATE} --guidance_scale 1.0 --network_type sd_no_bg --batch_size 8 -c 1 --feature_type clip --shadings_dir shadings --backgrounds_dir images --images_dir images -dataset /data/pakkapon/datasets/laion-shading/v3/train -dataset_split /data/pakkapon/datasets/laion-shading/v3/train/index/150k.json -dataset_val /data/pakkapon/datasets/laion-shading/v3/test -dataset_val_split /data/pakkapon/datasets/laion-shading/v3/test/index/main_v1.json  --val_check_interval 0.2 -specific_prompt _"

echo ${COMMAND}

singularity exec --bind /ist:/ist  --bind /ist/ist-share/vision/relight/datasets:/data/pakkapon/datasets --nv --env HF_HUB_CACHE=/ist/ist-share/vision/huggingface/hub/ --env HUB_HOME=/ist/ist-share/vision/huggingface/ ${CONTAINER} ${COMMAND}
