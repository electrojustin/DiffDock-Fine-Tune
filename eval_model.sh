#!/bin/bash

source /ext3/env.sh
conda activate $CONDA_NAME
python evaluate_alloset.py --config default_inference_args.yaml --model_dir ./workdir/test_score/${RUN_NAME} --ckpt best_ema_inference_epoch_model.pt --cache_path eval_cache_$1 --confidence_cache_path confidence_cache_$1 --split_path $EVAL_SPLIT_PATH --batch_size 10 --data_dir $DATASET_PATH --tqdm --chain_cutoff 10 --dataset alloset --protein_file protein --out_dir $RESULTS_DIR --esm_embeddings_path $ESM_EMBEDDINGS_PATH --num_dataloader_workers 12 --dataloader_prefetch_factor 100 --slurm_array_idx $1 --slurm_array_task_count $2
rm -rf eval_cache_$1
rm -rf confidence_cache_$1
