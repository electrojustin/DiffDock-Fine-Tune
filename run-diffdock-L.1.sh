#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
python evaluate_alloset.py --config default_inference_args.yaml --model_dir workdir/cb_test --ckpt last_model.pt --split_path /vast/eac709/allo_crossdock/splits/all_241120.txt --batch_size 10 --data_dir /vast/eac709/allo_crossdock/dataset_241120/ --tqdm --chain_cutoff 10 --dataset alloset --protein_file protein --out_dir ./results --smile_file ../../preprocessed_ligs/smiles.txt --esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --slurm_array_idx $1 --slurm_array_task_count $2
#python evaluate_alloset.py --config default_inference_args.yaml --model_dir ./workdir/test_score/allo-only --split_path /vast/eac709/allo_crossdock/splits/all_241120.txt --batch_size 10 --data_dir /vast/eac709/allo_crossdock/dataset_241120/ --tqdm --chain_cutoff 10 --dataset alloset --protein_file protein --out_dir ./results --smile_file ../../preprocessed_ligs/smiles.txt --esm_embeddings_path /vast/eac709/allo_crossdock/DiffDock-L/AlloSet/241120/data/esm2/ --slurm_array_idx $1 --slurm_array_task_count $2
