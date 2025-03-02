#!/bin/bash

#SBATCH --tasks 4
#SBATCH --cpus-per-task 4
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeg10045@nyu.edu
#SBATCH --mem=60G
#SBATCH --gres=gpu:4
#SBATCH --job-name=train-diffdock

module purge

source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

module load cuda/11.6.2

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
echo "MASTER_ADDR="$MASTER_ADDR

srun singularity exec --nv --overlay /scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-24.04.sif ./train-diffdock.sh
