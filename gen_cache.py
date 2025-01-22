import copy
import math
import os
import shutil
from functools import partial
import pickle

import torch
from tqdm import tqdm
import torch.distributed as dist
from socket import gethostname
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data.dataloader import DataLoader
from datasets.lazy_pdbbind import LazyPDBBindSet

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, t_to_sigma_individual
from datasets.loader import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, loss_function_ddp, inference_epoch_fix
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage

args = parse_train_args()
t_to_sigma = partial(t_to_sigma_compl, args=args)

cache_idx_path = os.path.join(args.cache_path, 'index.pkl')
cache_path = os.path.join(args.cache_path, 'cache.dat')
os.makedirs(args.cache_path, exist_ok=True)
try:
    os.remove(cache_idx_path)
except:
    pass
try:
    os.remove(cache_path)
except:
    pass

idx = {}

common_args = {'transform': None, 'limit_complexes': args.limit_complexes,
               'chain_cutoff': args.chain_cutoff, 'receptor_radius': args.receptor_radius,
               'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
               'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
               'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
               'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
               'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
               'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
               'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
               'matching_tries': args.matching_tries}

train_dataset = LazyPDBBindSet(ligand_file='fixed_ligand', cache_path=args.cache_path, split_path=args.split_train, keep_original=True, num_conformers=args.num_conformers, root=args.pdbbind_dir, esm_embeddings_path=args.pdbbind_esm_embeddings_path, protein_file=args.protein_file, **common_args)
val_dataset = LazyPDBBindSet(ligand_file='fixed_ligand', cache_path=args.cache_path, split_path=args.split_val, keep_original=True, num_conformers=args.num_conformers, root=args.pdbbind_dir, esm_embeddings_path=args.pdbbind_esm_embeddings_path, protein_file=args.protein_file, require_ligand=True, **common_args)

train_loader = DataLoader(prefetch_factor=args.dataloader_prefetch_factor, dataset=train_dataset, batch_size=1, num_workers=args.num_dataloader_workers, collate_fn=lambda x: x)
val_loader = DataLoader(prefetch_factor=args.dataloader_prefetch_factor, dataset=val_dataset, batch_size=1, num_workers=args.num_dataloader_workers, collate_fn=lambda x: x)


with open(cache_path, 'wb') as cache_file:
    print('Processing training data')
    for data in tqdm(train_loader, total=len(train_loader)):
        if not data:
            continue
        data = data[0]
        name = data.name
        offset = cache_file.tell()
        cache_file.write(pickle.dumps(data))
        size = cache_file.tell() - offset
        idx[name] = (offset, size)

    print('Processing validation data')
    for data in tqdm(val_loader, total=len(val_loader)):
        if not data:
            continue
        data = data[0]
        name = data.name
        offset = cache_file.tell()
        cache_file.write(pickle.dumps(data))
        size = cache_file.tell() - offset
        idx[name] = (offset, size)

with open(cache_idx_path, 'wb') as idx_file:
    pickle.dump(idx, idx_file)
