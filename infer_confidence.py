import functools
import logging
import pprint
import traceback
from argparse import ArgumentParser, Namespace, FileType
import copy
import os
from functools import partial
import warnings
from typing import Mapping, Optional
from rdkit.Chem import MolFromSmiles
from datasets.process_mols import read_molecule

import yaml

# Ignore pandas deprecation warning around pyarrow
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="(?s).*Pyarrow will become a required dependency of pandas.*")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs

# TODO imports are a little odd, utils seems to shadow things
from utils.logging_utils import configure_logger, get_logger
import utils.utils
from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.diffusion_utils import set_time
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm

if os.name != 'nt':  # The line does not work on Windows
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`")

# Prody logging is very verbose by default
prody_logger = logging.getLogger(".prody")
prody_logger.setLevel(logging.ERROR)

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default='default_inference_args.yaml')
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
    parser.add_argument('--complex_dir', type=str, default=None, help='Path to directory of complex files. Assumes the existence of a manifest file called "complex_names"')
    parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
    parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
    parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
    parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
    parser.add_argument('--pose_path', type=str, default=None, help='Path to file containing the final pose output by DiffDock')

    parser.add_argument('-l', '--log', '--loglevel', type=str, default='WARNING', dest="loglevel",
                        help='Log level. Default %(default)s')

    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

    parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=10, help='')
    parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')

    parser.add_argument('--old_score_model', action='store_true', default=False, help='')
    parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
    parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')
    parser.add_argument('--choose_residue', action='store_true', default=False, help='')

    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

    parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
    parser.add_argument('--gnina_path', type=str, default='gnina', help='')
    parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
    parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
    parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
    parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)
    parser.add_argument('--DDP', action='store_true', default=False)

    return parser


def get_pose_mol(pose_path):
    mol = MolFromSmiles(pose_path)
    if not mol:
        mol = read_molecule(pose_path, remove_hs=False, sanitize=True)
    return mol


def main(args):

    configure_logger(args.loglevel)
    logger = get_logger()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    # Download models if they don't exist locally
    if not os.path.exists(args.model_dir):
        logger.info(f"Models not found. Downloading")
        remote_urls = REMOTE_URLS
        downloaded_successfully = False
        for remote_url in remote_urls:
            try:
                logger.info(f"Attempting download from {remote_url}")
                files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
                if not files_downloaded:
                    logger.info(f"Download from {remote_url} failed.")
                    continue
                logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                downloaded_successfully = True
                # Once we have downloaded the models, we can break the loop
                break
            except Exception as e:
                pass

        if not downloaded_successfully:
            raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))
            confidence_args.DDP = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"DiffDock will run on {device}")

    if args.complex_dir is not None:
        complex_name_list = []
        with open(args.complex_dir + '/complex_names', 'r') as manifest_file:
            complex_name_list = list(map(lambda x: x.strip(), manifest_file.readlines()))
        protein_path_list = list(map(lambda x: args.complex_dir + '/' + x + '/' + x + '_protein_processed.pdb', complex_name_list))
        ligand_description_list = list(map(lambda x: args.complex_dir + '/' + x + '/' + x + '_ligand.sdf', complex_name_list))
        protein_sequence_list = None
        pose_path_list = ligand_description_list
    elif args.protein_ligand_csv is not None:
        df = pd.read_csv(args.protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist())
        protein_path_list = set_nones(df['protein_path'].tolist())
        protein_sequence_list = set_nones(df['protein_sequence'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())
        pose_path_list = set_nones(df['pose_path'].tolist())
    else:
        complex_name_list = [args.complex_name if args.complex_name else f"complex_0"]
        protein_path_list = [args.protein_path]
        protein_sequence_list = [args.protein_sequence]
        ligand_description_list = [args.ligand_description]
        pose_path_list = [args.pose_path]

    pose_mols = list(map(lambda x: get_pose_mol(x), pose_path_list))
    pose_mol_lookup = {}
    for i in range(0, len(complex_name_list)):
        pose_mol_lookup[complex_name_list[i]] = pose_mols[i]

    complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
    for name in complex_name_list:
        write_dir = f'{args.out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)

    logger.info('Confidence model uses different type of graphs than the score model. '
                'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = \
        InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                         ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                         lm_embeddings=True,
                         receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                         c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                         all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                         atom_max_neighbors=confidence_args.atom_max_neighbors,
                         knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                 confidence_mode=True, old=args.old_confidence_model)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()

    failures, skipped = 0, 0
    N = args.samples_per_complex
    test_ds_size = len(confidence_test_dataset)
    logger.info(f'Size of test dataset: {test_ds_size}')
    confidence_loader = DataLoader(dataset=confidence_test_dataset, batch_size=1, shuffle=False)
    with open('scores.tsv', 'w') as out_file:
        for idx, complex_graph_batch in tqdm(enumerate(confidence_loader)):
            try:
                complex_graph_batch = complex_graph_batch.to(device)
                set_time(complex_graph_batch, 0, 0, 0, 0, complex_graph_batch.num_graphs, confidence_args.all_atoms, device)
                pose_mol = pose_mol_lookup[complex_graph_batch.name[0]]
                complex_graph_batch.mol = pose_mol
                complex_graph_batch['ligand'].pos = torch.from_numpy(pose_mol.GetConformer().GetPositions()).float().to(device) - complex_graph_batch.original_center
                with torch.no_grad():
                    pred = confidence_model(complex_graph_batch)
                out_file.write(complex_graph_batch.name[0] + '\t' + str(float(pred[0][0])) + '\t' + str(float(pred[0][1])) + '\n')
            except Exception as e:
                logger.warning("Failed on", complex_graph_batch["name"], e)
                failures += 1

    result_msg = f"""
    Failed for {failures} / {test_ds_size} complexes.
    Skipped {skipped} / {test_ds_size} complexes.
"""
    if failures or skipped:
        logger.warning(result_msg)
    else:
        logger.info(result_msg)
    logger.info(f"Results saved in {args.out_dir}")


if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args)
