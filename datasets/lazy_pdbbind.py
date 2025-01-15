import binascii
import glob
import math
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import read_molecule, get_lig_graph_with_matching, generate_conformer, moad_extract_receptor_structure
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt, crop_beyond
from utils import so3, torus


class LazyPDBBindSet(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0, chain_cutoff=10,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 include_miscellaneous_atoms=False,
                 protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,
                 protein_file="protein_processed", ligand_file='ligand',
                 smile_file=None,
                 slurm_array_idx=None,
                 slurm_array_task_count=None,
                 knn_only_graph=False, matching_tries=1, dataset='AlloSet'):

        super(LazyPDBBindSet, self).__init__(root, transform)
        self.smile_file = smile_file
        self.ligand_smiles = {}
        self.slurm_array_idx = slurm_array_idx
        self.slurm_array_task_count = slurm_array_task_count
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.ligand_file = ligand_file
        self.dataset = dataset
        assert knn_only_graph or (not all_atoms)
        self.all_atoms = all_atoms
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'{dataset}3_limit{self.limit_complexes}'
                                                        # f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}' # temp change
                                                        f'_INDEXall_241120'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                                        f'_chainCutoff{self.chain_cutoff if self.chain_cutoff is None else int(self.chain_cutoff)}'
                                            + (''if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + (''if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + '_full'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))
                                            + ('' if protein_file == "protein_processed" else '_' + protein_file)
                                            + ('' if not self.fixed_knn_radius_graph else (f'_fixedKNN' if not self.knn_only_graph else '_fixedKNNonly'))
                                            + ('' if not self.include_miscellaneous_atoms else '_miscAtoms')
                                            + ('' if self.use_old_wrong_embedding_order else '_chainOrd')
                                            + ('' if self.matching_tries == 1 else f'_tries{matching_tries}'))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.preprocessing()

    def len(self):
        return len(self.complex_names_all)

    def get(self, idx):
        name = self.complex_names_all[idx]
        return self.get_by_name(name)

    def get_by_name(self, name):
        if not name or name not in self.complex_lm_embeddings:
            return None
        lm_embedding_chains = self.complex_lm_embeddings[name]
        complex_graph, lig = self.get_complex(name, lm_embedding_chains)
        if not complex_graph or (self.require_ligand and not lig):
            return None

        if self.require_ligand:
            complex_graph.mol = RemoveAllHs(lig)

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        return complex_graph

    def preprocessing(self):
        if self.smile_file:
            with open(self.smile_file, 'r') as smile_file:
                for row in smile_file.readlines():
                    parsed_row = row.split('\t')
                    self.ligand_smiles[(parsed_row[0].upper(), parsed_row[1].upper())] = parsed_row[2].strip()

        self.complex_names_all = read_strings_from_txt(self.split_path)
        if self.slurm_array_idx and self.slurm_array_task_count:
            slurm_task_size = int(math.ceil(len(self.complex_names_all) / self.slurm_array_task_count))
            start_idx = self.slurm_array_idx * slurm_task_size
            end_idx = min(len(self.complex_names_all), (self.slurm_array_idx + 1) * slurm_task_size)
            self.complex_names_all = self.complex_names_all[start_idx:end_idx]
            print('Processing complexes ' + str(start_idx) + ' through ' + str(end_idx))
        else:
            if self.limit_complexes is not None and self.limit_complexes != 0:
                self.complex_names_all = complex_names_all[:self.limit_complexes]
        # generate embeddings for all of the complexes up front
        # load only the embeddings specific to the test set
        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                key_name = key.split('_chain_')[0]
                if key_name in self.complex_names_all:
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_indices_dictlist[key_name].append(int(key.split('_chain_')[1]))
            self.lm_embeddings_chains_all = []
            for name in self.complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = [complex_chains_embeddings[i] for i in chain_reorder_idx]
                self.lm_embeddings_chains_all.append(reordered_chains)
        else:
            self.lm_embeddings_chains_all = [None] * len(self.complex_names_all)
        self.complex_lm_embeddings = {}
        for i in range(0, len(self.complex_names_all)):
            self.complex_lm_embeddings[self.complex_names_all[i]] = self.lm_embeddings_chains_all[i]

    def get_complex(self, name, lm_embedding_chains):
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)):
            print("Folder not found", name)
            return None, None,

        try:
            parsed_name = name.split('_')
            pdb = parsed_name[0]
            lig_name = parsed_name[2]
            orig_lig_pos = None
            if self.ligand_smiles and (pdb.upper(), lig_name.upper()) in self.ligand_smiles:
                lig_smiles = self.ligand_smiles[(pdb.upper(), lig_name.upper())]
                lig = Chem.MolFromSmiles(lig_smiles)
                generate_conformer(lig)
                Chem.SanitizeMol(lig)
                lig = Chem.RemoveHs(lig, sanitize=True)
                orig_lig_pos = read_mol(self.pdbbind_dir, name, pdb, suffix=self.ligand_file, remove_hs=True).GetConformers()[0].GetPositions()
                if orig_lig_pos is None:
                    print('Error loading ligand original atom positions')
                    return None, None
            else:
                lig = read_mol(self.pdbbind_dir, name, pdb, suffix=self.ligand_file, remove_hs=False)


            if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                return None, None
            complex_graph = HeteroData()
            complex_graph['name'] = name
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries)

            moad_extract_receptor_structure(path=os.path.join(self.pdbbind_dir, name, f'{pdb}_{self.protein_file}.pdb'),
                                            complex_graph=complex_graph,
                                            neighbor_cutoff=self.receptor_radius,
                                            max_neighbors=self.c_alpha_max_neighbors,
                                            lm_embeddings=lm_embedding_chains,
                                            knn_only_graph=self.knn_only_graph,
                                            all_atoms=self.all_atoms,
                                            atom_cutoff=self.atom_radius,
                                            atom_max_neighbors=self.atom_max_neighbors)
            if orig_lig_pos is not None:
                complex_graph['ligand'].orig_pos = orig_lig_pos

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return None, None

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center
        complex_graph['receptor_name'] = name
        return complex_graph, lig

def read_mol(pdbbind_dir, complex_name, pdb_name, suffix='ligand', remove_hs=False):
    try:
        lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.sdf'), remove_hs=remove_hs, sanitize=True)
    except:
        lig = None
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        try:
            lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.mol2'), remove_hs=remove_hs, sanitize=True)
        except:
            lig = None
    if lig is None:  # read pdb file if neither sdf nor mol2 can be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.pdb'), remove_hs=remove_hs, sanitize=True)
    return lig
