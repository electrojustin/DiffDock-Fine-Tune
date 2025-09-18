#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock

# --pretrain_dir='workdir/v1.1/score_model/' --pretrain_ckpt='best_ema_inference_epoch_model'

python train.py --config default_inference_args.yaml --protein_file protein --pretrain_dir='workdir/v1.1/score_model/' --pretrain_ckpt='best_ema_inference_epoch_model' --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train allo_train_split.txt --split_val allo_val_split.txt --val_inference_freq 100 --train_inference_freq 100 --n_epochs 5000 --num_dataloader_workers 4 --pin_memory --DDP --num_inference_complexes 50 --wandb --entity jeg10045-new-york-university --project diffdock-tune --run_name weighted_tor_lin_fixed --lr 0.0001 --weighted_tor 6

#python train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train allo_train_split.txt --split_val allo_val_split.txt --restart_dir='workdir/test_score/weighted-tor3' --val_inference_freq 100 --train_inference_freq 100 --n_epochs 5000 --num_dataloader_workers 4 --pin_memory --DDP --num_inference_complexes 50 --wandb --entity jeg10045-new-york-university --project diffdock-tune --run_name weighted-tor3 --lr 0.0001 --weighted_tor 3 --wandb_id 6qcf4asm

#python train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train debug_train_split.txt --split_val debug_test_split.txt --val_inference_freq 100 --train_inference_freq 100 --n_epochs 10000 --num_dataloader_workers 4 --pin_memory --DDP --num_inference_complexes 50 --wandb --entity jeg10045-new-york-university --project hyperparam --run_name weighted_tor2 --lr 0.0001 --pretrain_dir='workdir/v1.1/score_model/' --pretrain_ckpt='best_ema_inference_epoch_model' --weighted_tor
