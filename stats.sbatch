#!/bin/bash

#SBATCH --cpus-per-task 1
#SBATCH --mem=10G

source config.sh

singularity exec --overlay ${OVERLAY_PATH}:ro $SIF_PATH ./stats.sh
