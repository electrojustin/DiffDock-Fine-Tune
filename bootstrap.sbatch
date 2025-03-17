#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --mem=60GB
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cb_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeg10045@nyu.edu

module load cuda/11.6.2
singularity exec --nv --overlay /scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-24.04.sif ./bootstrap.sh
