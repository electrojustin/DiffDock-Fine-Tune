#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
python train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train allo_train_split.txt --split_val allo_test_split.txt --val_inference_freq 10 --pdbbind_inference_freq 10 --n_epochs 100 --pin_memory --DDP --pretrain_dir='workdir/v1.1/score_model' --pretrain_ckpt='best_ema_inference_epoch_model'
#python train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 100 --pin_memory --DDP 
#python -m cProfile -s cumtime train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --pin_memory --DDP --limit_complexes 100
