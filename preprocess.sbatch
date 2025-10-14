#!/bin/bash

#SBATCH --cpus-per-task 48
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --job-name=cache-hetgraphs

source config.sh

module load $OBABEL_MODULE_NAME

singularity exec --nv --overlay ${OVERLAY_PATH}:ro $SIF_PATH ./preprocess.sh
