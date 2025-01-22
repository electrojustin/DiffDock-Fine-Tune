#!/bin/bash

source /ext3/env.sh
conda activate diffdock-pocket
cd /scratch/projects/yzlab/group/kinase_eric/kinase_location_classifier/jeg10045_kinase_scripts/diffdock/DiffDock
python train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1000 --pin_memory --DDP 
#python -m cProfile -s cumtime train.py --config default_inference_args.yaml --protein_file protein --pdbbind_esm_embeddings_path data/esm_embedding_output --n_epochs 1 --pin_memory --DDP --limit_complexes 100
