#!/bin/bash

#SBATCH --tasks 4
#SBATCH --cpus-per-task 4
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --job-name=train-diffdock

source config.sh

module purge

module load $CUDA_MODULE_NAME

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
echo "MASTER_ADDR="$MASTER_ADDR

srun singularity exec --nv --overlay ${OVERLAY_PATH}:ro $SIF_PATH ./train_diffdock.sh
