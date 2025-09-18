#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock

python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train ../../../final_allo_only_splits_250604/train.txt --split_val ../../../final_allo_only_splits_250604/val_x4.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100 --weighted_tor 7

#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100
#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes_esm/ --pdbbind_dir /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes_selRMSDlt2 --split_train /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/kinase_md/supersampled_complexes/complex_names.txt --split_val ../../../final_allo_only_splits_250604/val_x4.txt --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100 --weighted_tor 7

#python gen_cache.py --config default_inference_args.yaml --smile_file data/PDBBind_processed/smiles.txt --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output/ --pdbbind_dir data/PDBBind_processed --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val --n_epochs 1 --log_dir workdir/delete_me --num_dataloader_workers 48 --dataloader_prefetch_factor 100 --weighted_tor 7

#python gen_cache.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --pdbbind_dir data/allosteric_dataset/ --split_train debug_train_split.txt --split_val debug_train_split.txt --n_epochs 1 --log_dir workdir/delete_me --weighted_tor 1


rm -rf workdir/delete_me
