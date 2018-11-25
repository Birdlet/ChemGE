import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from rdkit import rdBase
import copy

import sascorer
import nltk
import numpy as np
from config import zinc_gram #import zinc_grammar


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if smiles != '' and mol is not None and mol.GetNumAtoms() > 1:
        return Chem.MolToSmiles(mol)
    else:
        return ''
