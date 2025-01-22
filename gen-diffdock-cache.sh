#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
#rm -rf data/cache_torsion/*
python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --batch_size 48 --dataloader_prefetch_factor 100
rm -rf workdir/delete_me
