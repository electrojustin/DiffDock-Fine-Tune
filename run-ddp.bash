#!/bin/bash

source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

#export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -1)
echo "MASTER_ADDR="$MASTER_ADDR

srun run-DD-sing.bash \
    python -m train --config=/scratch/eac709/overlays/DiffDock-DDP/241022/diffdock-s_training_args.yaml \
        --cache_path=/scratch/eac709/overlays/DiffDock-DDP/241022/cache.1/ \
        --log_dir=/scratch/eac709/overlays/DiffDock-DDP/241022/log_2gpu/ \
        --limit_complexes=-1 \
        --n_epochs=10 --batch_size=16 --num_dataloader_workers=1 --pin_memory --DDP \
        --run_name=2rtx8000_bs64_dw1_pm_DDP
