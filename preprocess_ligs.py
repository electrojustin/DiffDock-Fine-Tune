from pathlib import Path
import os
import traceback
import glob
import csv
import requests
import rdkit
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

working_dir = os.getcwd()

for in_file in glob.glob(sys.argv[1] + '/*/*_ligand.*'):
  if 'fixed' in in_file:
    continue
  dir_name = os.path.dirname(in_file)
  base_name = os.path.basename(in_file)
  print('Processing ' + base_name)
  os.chdir(dir_name)

  try:
    if os.system('obabel ' + base_name + ' -O lig_h.sdf -p 7.4'):
      break
  
    lig_mol = Chem.MolFromMolFile('lig_h.sdf', removeHs=False, sanitize=False)

    # Handle all bugs that require hydrogen regeneration.

    # Obabel doesn't handle imidazoles correctly all the time. Sometimes it just removes all hydrogens and makes a pentavalent carbon.
    lig_atoms = list(lig_mol.GetAtoms())
    needs_reprotonation = False
    for error in Chem.DetectChemistryProblems(lig_mol):
      atom = lig_atoms[error.GetAtomIdx()]
      if atom.GetAtomicNum() == 6 and atom.GetTotalValence() == 5:
        num_double_nitrogen = 0
        for bond in atom.GetBonds():
          if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetAtomicNum() == 7:
            num_double_nitrogen += 1
        if num_double_nitrogen == 2:
          # Imidazole glitch detected
          needs_reprotonation = True
          for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetAtomicNum() == 7:
              # Arbitrarily pick one imidazole nitrogen to make a single bond
              bond.SetBondType(Chem.rdchem.BondType.SINGLE)
              break
    if needs_reprotonation:
      Chem.MolToMolFile(lig_mol, 'lig_fixed.sdf')
      if os.system('obabel lig_fixed.sdf -O lig_h.sdf -p 7.4'):
        break
      lig_mol = Chem.MolFromMolFile('lig_h.sdf', removeHs=False, sanitize=False)

    # Handle all bugs that require structural changes to the molecule

    lig_atoms = list(lig_mol.GetAtoms())
    needs_write = False
    # Obabel is sometimes overzealous and protonates quaternary amines, resulting in pentavalent nitrogen
    for error in Chem.DetectChemistryProblems(lig_mol):
      atom = lig_atoms[error.GetAtomIdx()]
      if atom.GetAtomicNum() == 7 and atom.GetTotalValence() == 5:
        for bond in atom.GetBonds():
          if bond.GetOtherAtom(atom).GetAtomicNum() == 1:
            # I couldn't figure out an easy way to just delete one atom, so I set the hydrogen charge to something impossible an delete all hydrogens with a crazy charge.
            bond.GetOtherAtom(atom).SetFormalCharge(-2)
            double_bonded_oxygen = []
            break
    # Obabel will generate double bonds between both oxygens in a nitro group. So we gotta cleave one and probably fix the formal charges.
    for atom in lig_mol.GetAtoms():
      if atom.GetAtomicNum() == 7:
        double_bonded_oxygen = []
        for bond in atom.GetBonds():
          # RDKit will silently correct the bond type even though the underlying SDF file has a pentavalent nitrogen, so just look for any nitrogen attached to two oxygens.
          if bond.GetOtherAtom(atom).GetAtomicNum() == 8:
            double_bonded_oxygen.append(bond)
        if len(double_bonded_oxygen) == 2:
          double_bonded_oxygen[0].GetOtherAtom(atom).SetFormalCharge(0)
          double_bonded_oxygen[0].SetBondType(Chem.rdchem.BondType.DOUBLE)
          double_bonded_oxygen[1].GetOtherAtom(atom).SetFormalCharge(-1)
          double_bonded_oxygen[1].SetBondType(Chem.rdchem.BondType.SINGLE)
          atom.SetFormalCharge(1)
    # Delete the bad hydrogens
    query = Chem.MolFromSmarts('[#1--]')
    lig_mol = AllChem.DeleteSubstructs(lig_mol, query)

    # Handle all bugs that just require adjusting the formal charge

    lig_atoms = list(lig_mol.GetAtoms())
    # RDKit doesn't always get the charges correct
    for error in Chem.DetectChemistryProblems(lig_mol):
      atom = lig_atoms[error.GetAtomIdx()]
      if atom.GetAtomicNum() == 7 and atom.GetTotalValence() == 4:
        # Protonated amines (and quaternary amines) should have a +1 charge, but RDKit sometimes gives them 0
        atom.SetFormalCharge(1)
      elif atom.GetAtomicNum() == 7 and atom.GetTotalValence() == 3:
        # Weird glitch where RDKit will give tetrazole nitrogens a -1 charge
        atom.SetFormalCharge(0)
      elif atom.GetAtomicNum() == 8: 
        bond = atom.GetBonds()
        if bond and bond[0].GetBondType() == Chem.rdchem.BondType.DOUBLE and atom.GetFormalCharge() == -1:
          # Glitch where rdkit will give both oxygens in a deprotonated carboxyl a negative charge, but keep the double bond for one of them
          atom.SetFormalCharge(0)
        elif len(bond) == 2 and bond[0].GetBondType() == Chem.rdchem.BondType.SINGLE and bond[1].GetBondType() == Chem.rdchem.BondType.SINGLE and atom.GetFormalCharge() == -1:
          # Glitch where rdkit will give an ether oxygen a negative charge for some reason?
          atom.SetFormalCharge(0)
      elif atom.GetAtomicNum() == 5:
        # Give up on boron chemistry for now.
        print('Cannot process boron chemistry!')
      else:
        print(atom.GetAtomicNum())
        print(atom.GetTotalValence())
        print(atom.GetFormalCharge())
        print('Chemistry problem! ' + str(error))
        break

    Chem.SanitizeMol(lig_mol)
    Chem.MolToMolFile(lig_mol, base_name.replace('ligand', 'fixed_ligand')[:-3] + 'sdf')
    with open('../smiles.txt', 'a') as out_file:
      out_file.write(dir_name[2:] + '\t' + Chem.MolToSmiles(lig_mol) + '\n')
  except:
    print('Failed to process ' + base_name)
    print(traceback.format_exc())

  os.chdir(working_dir)

print('Last processed: ' + dir_name)
