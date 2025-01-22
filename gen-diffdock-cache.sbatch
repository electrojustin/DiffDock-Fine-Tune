#!/bin/bash

#SBATCH --cpus-per-task 3
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeg10045@nyu.edu
#SBATCH --mem=60G
#SBATCH --job-name=cache-hetgraphs

singularity exec --nv --overlay /scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-24.04.sif ./gen-diffdock-cache.sh
