#!/bin/bash

#SBATCH --cpus-per-task 48
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jeg10045@nyu.edu
#SBATCH --mem=60G
#SBATCH --job-name=cache-hetgraphs

source config.sh

singularity exec --nv --overlay ${OVERLAY_PATH}:ro $SIF_PATH ./preprocess.sh
