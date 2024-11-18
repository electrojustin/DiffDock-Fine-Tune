#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done

if [ "$args" == "" ]; then args="/bin/bash"; fi

source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
	--overlay /scratch/eac709/overlays/diffdock-L.py39.lightning.241030.ext3:ro \
	--overlay /scratch/work/public/singularity/slurm-23.11.4.sqf:ro \
	/scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
${args} 
"