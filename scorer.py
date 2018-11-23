import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import rdBase

import sascorer
import os
import subprocess
import hashlib

rdBase.DisableLog('rdApp.error')
INFINITY = 1e4

# from https://github.com/gablg1/ORGAN/blob/master/organ/mol_metrics.py#L83
def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

# from grammar VAE
# logP_values = np.loadtxt('logP_values.txt')
# SA_scores = np.loadtxt('SA_scores.txt')
# cycle_scores = np.loadtxt('cycle_scores.txt')
logP_mean = 2.457    # np.mean(logP_values)
logP_std = 1.434     # np.std(logP_values)
SA_mean = -3.053     # np.mean(SA_scores)
SA_std = 0.834       # np.std(SA_scores)
cycle_mean = -0.048  # np.mean(cycle_scores)
cycle_std = 0.287    # np.std(cycle_scores)


def score(smiles):
    if verify_sequence(smiles):
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if Descriptors.MolWt(molecule) > 500:
                return -INFINITY
            current_log_P_value = Descriptors.MolLogP(molecule)
            current_SA_score = -sascorer.calculateScore(molecule)
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(molecule)))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            current_cycle_score = -cycle_length

            current_SA_score_normalized = (current_SA_score - SA_mean) / SA_std
            current_log_P_value_normalized = (current_log_P_value - logP_mean) / logP_std
            current_cycle_score_normalized = (current_cycle_score - cycle_mean) / cycle_std

            score = (current_SA_score_normalized
                     + current_log_P_value_normalized
                     + current_cycle_score_normalized)
            return score
        except Exception:
            return -INFINITY
    else:
        return -INFINITY


def score_rdock(smiles, num_docking=3):
    smiles_md5 = str(hashlib.md5(smiles.encode('utf-8')).hexdigest())
    docking_result_file = '{}_out'.format(smiles_md5)
    sdf_name = '{}.sdf'.format(smiles_md5)
    score_name = '<SCORE.INTER>'  # <SCORE> or <SCORE.INTER>

    min_score = INFINITY

    # Translation from SMILES to sdf
    if smiles == '':
        mol = None
    else:
        mol = Chem.MolFromSmiles(smiles)
    try:
        if mol is not None and Descriptors.MolWt(mol) < 500:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            fw = Chem.SDWriter(sdf_name)
            fw.write(mol)
            fw.close()

            # rdock calculation
            cmd = '$RBT_ROOT/bin/rbdock -r cavity.prm '\
                  '-p $RBT_ROOT/data/scripts/dock.prm '\
                  '-i {} -o {} -T 1 -n {} > /dev/null'\
                  .format(sdf_name, docking_result_file, num_docking)
            path = docking_result_file+'.sd'
            if not os.path.exists(path):
                subprocess.call(cmd, shell=True)

            # find the minimum score of rdock from multiple docking results
            if os.path.exists(path):
                with open(path, 'r') as f:
                    lines = f.readlines()
                isScore = False
                for line in lines:
                    if isScore:
                        min_score = min(float(line), min_score)
                        isScore = False
                    if score_name in line:  # next line has score
                        isScore = True
    except Exception:
        pass
    return min_score

def _vina(vinalog):
    with open(vinalog) as f:
        lines = f.read().split("\n")
        start = lines.index("-----+------------+----------+----------")+1
        try:
            score = float(lines[start].split()[1].strip())
        except:
            score = INFINITY
    return score


def score_vina(smiles, conf_path, log_path='', num_docking=3, cpu=4):
    smiles_md5 = str(hashlib.md5(smiles.encode('utf-8')).hexdigest())

    pdb_name = os.path.join(log_path, '{}.pdb'.format(smiles_md5))
    pdbqt_name = os.path.join(log_path, '{}.pdbqt'.format(smiles_md5))
    docking_result_file = os.path.join(log_path, '{}.log'.format(smiles_md5))

    min_score = INFINITY

    # Translation from SMILES to sdf
    if smiles == '':
        mol = None
    else:
        mol = Chem.MolFromSmiles(smiles)
    try:
        if mol is not None and Descriptors.MolWt(mol) < 500:
            # save ligand as .mol2 format
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            Chem.MolToPDBFile(mol, pdb_name)

            # vina ligand preparing
            cmd = '/opt/ADT/MGLTools-1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'\
                    ' -l {} -o {} > /dev/null'\
                    .format(pdb_name, pdbqt_name)
            subprocess.call(cmd, shell=True)

            # vina dock processing
            cmd = '/opt/ADT/vina/bin/qvina --config {} '\
                  '--ligand {} '\
                  '--log {} --cpu {} > /dev/null'\
                  .format(conf_path, pdbqt_name, docking_result_file, cpu)
            subprocess.call(cmd, shell=True)

            # find the minimum score of rdock from multiple docking results
            min_score = min(_vina(docking_result_file), INFINITY)
    except Exception:
        pass
        #raise Exception
    return min_score
