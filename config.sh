#!/bin/bash

# Sets up slurm singularity env variables on the NYU HPC.
# This should be commented out (or pointed to a new path) for non-NYU users.
source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

###############################################################################
######## These variables will generally need to be changed by everyone ########
###############################################################################

# Name of the slurm module that provides CUDA bindings.
export CUDA_MODULE_NAME=cuda/11.6.2
# Name of the slurm module that provides OpenBabel.
export OBABEL_MODULE_NAME=openbabel/intel/3.1.1
# Path to the singularity container where DiffDock dependencies were installed.
export OVERLAY_PATH=/scratch/jeg10045/zhang_lab/overlay-50G-10M.ext3
# Path to singularity image file.
export SIF_PATH=/scratch/work/public/singularity/ubuntu-24.04.sif
# Path to the directory where the ESM embeddings were deposited.
export ESM_EMBEDDINGS_PATH=cleanup_test_dataset_esm
# Path to dataset. This is expected to be a directory containing subdirectories
# for each complex named in the form PDBID_CHAIN_LIGNAME. For example,
# 7wcl_A_8ZF. Each subdirectory is expected to contain two files: an apo
# protein PDB file named PDBID_protein.pdb, and a ligand PDF file named
# PDBID_ligand.pdb.
export DATASET_PATH=cleanup_test_dataset
# File containing a list of the complexes used for training. These complexes
# follow the same naming scheme as the subdirectories in the dataset directory.
export TRAIN_SPLIT_PATH=../../../final_allo_only_splits_250604/train.txt
# File containg a list of the complexes used for validation. These complexes
# follow the same naming scheme as the subdirectories in the dataset directory,
# and the complexes in the training split.
export VAL_SPLIT_PATH=../../../final_allo_only_splits_250604/val_x4.txt
# Maximum number of epochs to train for.
export NUM_EPOCHS=5000
# WANDB entity name. Leave blank to disable WANDB.
export WANDB_ENTITY=jeg10045-new-york-university
# WANDB project name. Leave blank if not using WANDB.
export WANDB_PROJECT=diffdock-tune
# Run name. This will both be the WANDB run name, and the name of the model.
# The fine tuned model will be put in a directory named
# workdir/test_score/$RUN_NAME
export RUN_NAME=cleanup_test
# List of complexes to run eval on. Complexes have the same naming convention
# as the TRAIN_SPLIT_PATH and VAL_SPLIT_PATH files.
export EVAL_SPLIT_PATH=../../../final_allo_only_splits_250604/test.txt
# Directory where the eval results will be stored. This will include both
# snapshots of the reverse diffusion process in PDB form and aggregate
# statistics stored as numpy files.
export RESULTS_DIR=./cleanup_results


###############################################################################
######### These variables can generally be left alone by most people ##########
###############################################################################

# Cache path for training.
export CACHE_PATH=data/cache
# Pretrain model directory.
export PRETRAIN_DIR='workdir/v1.1/score_model/'
# Name of the checkpoint used as the pretrain model.
export PRETRAIN_CKPT='best_ema_inference_epoch_model'
# Name of the conda environment where DiffDock dependencies were installed.
export CONDA_NAME=diffdock-pocket
