#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=AlloSet_DD_L
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeg10045@nyu.edu

module load cuda/11.6.2
singularity exec --nv --overlay /scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-24.04.sif ./run-diffdock-L.1.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
