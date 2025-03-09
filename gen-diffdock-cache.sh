#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train allo_train_split.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 12 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes_esm/embeddings/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val allo_test_split.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 1 --dataloader_prefetch_factor 100
rm -rf workdir/delete_me
