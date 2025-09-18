#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
python train.py --config default_inference_args.yaml --pretrain_dir='workdir/v1.1/score_model/' --pretrain_ckpt='best_ema_inference_epoch_model' --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train ../../../final_allo_only_splits_250604/train.txt --split_val ../../../final_allo_only_splits_250604/val_x4.txt --val_inference_freq 100 --train_inference_freq 100 --n_epochs 5000 --num_dataloader_workers 4 --pin_memory --DDP --num_inference_complexes 100 --wandb --entity jeg10045-new-york-university --project diffdock-tune --run_name allo_only --lr 0.0001 --inference_earlystop_avg_infsteps=10
