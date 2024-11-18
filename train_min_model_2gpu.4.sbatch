#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=2rtx8000_bs64_dw4_pm_DDP
#SBATCH --partition=rtx8000
###SBATCH --partition=chemistry_a100_2,stake_a100_1,stake_a100_2,a100_1,a100_2,v100,rtx8000
### ntasks-per-node=<n> must equal gpu:<n> 

module purge

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
        --n_epochs=50 --batch_size=64 --num_dataloader_workers=4 --pin_memory --DDP \
        --run_name=2rtx8000_bs64_dw4_pm_DDP --wandb 

# srun singularity exec --nv --overlay /scratch/eac709/overlays/diffdock-L.py39.lightning.241030.ext3:ro /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "
# source /ext3/env.sh ;
# python -m train --config=/scratch/eac709/overlays/DiffDock-DDP/241022/diffdock-s_training_args.yaml \
#     --cache_path=/scratch/eac709/overlays/DiffDock-DDP/241022/cache.1/ \
#     --log_dir=/scratch/eac709/overlays/DiffDock-DDP/241022/log_4gpu/ \
#     --limit_complexes=-1 \
#     --n_epochs=50 --batch_size=64 --num_dataloader_workers=4 --pin_memory --DDP \
#     --wandb --run_name=2rtx8000_bs64_dw4_pm_DDP
# "
