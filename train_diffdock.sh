#!/bin/bash

source /ext3/env.sh
conda activate $CONDA_NAME
python train.py --config default_inference_args.yaml --pretrain_dir=${PRETRAIN_DIR} --pretrain_ckpt=${PRETRAIN_CKPT} --protein_file protein --pdbbind_esm_embeddings_path $ESM_EMBEDDINGS_PATH --pdbbind_dir $DATASET_PATH --split_train $TRAIN_SPLIT_PATH --split_val $VAL_SPLIT_PATH --val_inference_freq 100 --train_inference_freq 100 --n_epochs $NUM_EPOCHS --num_dataloader_workers 4 --pin_memory --DDP --num_inference_complexes 100 --wandb --entity $WANDB_ENTITY --project $WANDB_PROJECT --run_name $RUN_NAME --lr 0.0001 --inference_earlystop_avg_infsteps=10
