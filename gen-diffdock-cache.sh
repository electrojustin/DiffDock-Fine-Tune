#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
rm -rf data/cache
python gen_cache.py --config default_inference_args.yaml --protein_file protein --cache_path data/cache --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train ../../../final_allo_only_splits_250604/train.txt --split_val ../../../final_allo_only_splits_250604/val_x4.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100

rm -rf workdir/delete_me
