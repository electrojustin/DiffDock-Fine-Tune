# DiffDock Fine Tuning 

This is a fork of DiffDock-L with performance optimizations and wrapper scripts designed to make fine-tuning easier.

The original DiffDock-L source can be found [here](https://github.com/gcorso/DiffDock/).

## System Requirements

The wrapper scripts in this repo are designed to run on an HPC using Slurm for job management and Singularity for container management. The Slurm installation must have a cuda module and an openbabel module.

The HPC these scripts were tested on have general compute nodes with 48+ CPU cores, and GPU nodes with 4 GPUs. Both types of nodes have 120+GB of general RAM available, and each GPU in the GPU nodes have 80GB of VRAM. If your HPC's general compute nodes have fewer CPUs per node, it may be worth editing `cpus-per-task` in `preprocess.sbatch` and `num_dataloader_workers` in `preprocess.sh` accordingly. If your HPC's GPU nodes have a different number of attached GPUs, it may be necessary to edit `tasks` and `gres` in `train_diffdock.sbatch` accordingly.

## Obtaining Pretrained Model

This repository is designed for fine-tuning an existing DiffDock-L model rather than training a whole new model from scratch, which can be very computationally expensive. The original authors of DiffDock-L have kindly published their model [here](https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip), which is suitable for this purpose.

## Environmental Setup

DiffDock requires a Singularity container with an appropriate Conda environment installed.

For instructions to setup a Conda environment inside of a Singularity container, see [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda), or check with your HPC administrator.

From inside the newly created Conda Singularity container, DiffDock's dependencies can be installed by running

    conda env create --file environment.yml
    conda activate diffdock

If using wandb.ai to track training progress, wandb can be set up by following just step 1 of the directions [here](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart&product=models).

## Dataset Preparation

These scripts have been written assuming a rather specific layout of the dataset being used for both training and testing. We assume that both training and testing data are co-located in the same directory, and the distrinction between training and testing is managed through "split files", which are just plain text documents containing lists of complexes. Complexes need to be named in the form `{PDB ID}_{CHAIN ID}_{LIGAND NAME}_{OPTIONAL SUFFIX}`. The dataset directory is assumed to contain sub-directories of each complex. Each sub-directory should contain 2 files: a protein file, and a ligand file. The protein files should contain the chain of the apo protein that actually binds to the ligand, while the ligand file should contain all the HETATM entries consisting of the ligand. The protein file should be named using the scheme `{PDB ID}_protein.pdb`, while the ligand file should be named like `{PDB ID}_ligand.pdb`. Each complex sub-directory should only contain one protein chain and one ligand, so higher order complexes need to be broken up accordingly. An example dataset directory might look like this:

    my_dataset/8ssp_A_627_510/8ssp_protein.pdb
    my_dataset/8ssp_A_627_510/8ssp_ligand.pdb
    my_dataset/8ssp_A_PDO_507/8ssp_protein.pdb
    my_dataset/8ssp_A_PDO_507/8ssp_ligand.pdb

## ESM Embedding Preparation

It's still necessary to generate ESM embeddings for all the proteins in the dataset. In order to do this, simply [follow the instructions outlined in the original DiffDock repo](https://github.com/gcorso/DiffDock/blob/main/README.md#generate-the-esm2-embeddings-for-the-proteins), *but skip the very last step, where it says to run esm_embeddings_to_pt.py*. This last step collates all the ESM embeddings in the dataset into one big file, which we have noticed can sometimes cause out of memory issues. Instead, we have modified the dataloader to process the ESM embeddings as individual files to keep the memory footprint down.

## Configuration

The last manual step in setting up DiffDock is to edit the configuration file, `config.sh`. This file contains a number of configuration variables (e.g. run names, input data paths, etc.) needed by downstream scripts. See the comments in `config.sh` for more details on what each variable means.

## Fine Tuning

### Preprocessing

While the training code can theoretically be run with no preprocessing step and will simply perform preprocessing on the fly in the dataloader thread, in practice, some ligands with a large number of rotatable bonds will take excessive amounts of CPU time in the conformer matching stage. In order to make the most efficient use of GPU node time, we've shuffled much of the CPU expensive operations into a discrete preprocessing step that can be run on a general compute node.

Our preprocessing scripts also take care of the problem of hydrogenation. Because most crystal structures lack hydrogens, there are some ambiguities in how the ligand file should be loaded, and RDKit, the library DiffDock uses for this purpose, often makes mistakes. In particular, RDKit doesn't seem readily able to distinguish between aromatic and cycloalkane rings in PDB files, which can have a profound impact on model performance. Our preprocessing scripts take care of this problem by explicitly adding hydrogens to the ligand and converting it to an SDF file. `See preprocess_ligs.py` for details on the algorithm.

Once `config.sh` is properly set up, preprocessing should be  as easy as running

    sbatch preprocess.sbatch

Note that this will attempt to launch a job on a 48 core general compute node. Please adjust `cpus-per-task` in `preprocess.sbatch` if this isn't feasible on your HPC.

### Training

After preprocessing is complete, training can be started by running

    sbatch train_diffdock.sbatch

Note that this will attempt to launch a job on a GPU node with 4 open GPUs that lasts 2 days. Please adjust `tasks` and `gres` in `train_diffdock.sbatch` if your HPC does not have GPU nodes with 4 GPUs. Both variables must match the number of GPUs. Note that reducing this number from 4 will likely make training take longer.

## Testing

### Running The Model

Unlike the preprocessing and training scripts, the model eval script is designed to be run as an array job, since model evalutation is easily parallelizable in a way that doesn't require communication between workers. To evaluate the model's performance on the test dataset, simply run

    sbatch --array=0-<number of workers> eval_model.sbatch

### Statistical Summary

The eval script will create a "results" directory (configured in `config.sh`) containing PDB files depicting all the reverse diffusion steps of each docking run. It will also contain accuracy statistics from each eval worker, which can be aggregated using

    sbatch stats.sbatch
