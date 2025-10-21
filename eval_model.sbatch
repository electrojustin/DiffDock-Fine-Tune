#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=eval_model
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12

source config.sh

module load $CUDA_MODULE_NAME
singularity exec --nv --overlay ${OVERLAY_PATH}:ro $SIF_PATH ./eval_model.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
