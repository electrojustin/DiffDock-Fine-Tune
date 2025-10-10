#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock

python evaluate_alloset.py --config default_inference_args.yaml --model_dir ./workdir/test_score/cleanup_test --ckpt best_ema_inference_epoch_model.pt --cache_path eval_cache_$1 --confidence_cache_path confidence_cache_$1 --split_path /vast/eac709/allo_crossdock/splits/all_241120.txt --batch_size 10 --data_dir data/cleanup_test_dataset --tqdm --chain_cutoff 10 --dataset alloset --protein_file protein --out_dir ./cleanup_results --esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --num_dataloader_workers 12 --dataloader_prefetch_factor 100 --slurm_array_idx $1 --slurm_array_task_count $2
rm -rf eval_cache_$1
rm -rf confidence_cache_$1
