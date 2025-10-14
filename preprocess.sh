#!/bin/bash

source /ext3/env.sh
conda activate $CONDA_NAME

python preprocess_ligs.py $DATASET_PATH

rm -rf $CACHE_PATH
python gen_cache.py --config default_inference_args.yaml --protein_file protein --cache_path $CACHE_PATH --pdbbind_esm_embeddings_path $ESM_EMBEDDINGS_PATH --pdbbind_dir $DATASET_PATH --split_train $TRAIN_SPLIT_PATH --split_val $VAL_SPLIT_PATH --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100

rm -rf workdir/delete_me
