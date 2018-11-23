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


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except Exception:
        return ''


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
        return smiles


def vina_score(vinalog, num_docking=3):
    scores = 0
    with open(vinalog) as f:
        lines = f.read().split("\n")
        start = lines.index("-----+------------+----------+----------")
        end = lines.index("Writing output ... done.")
        if end-start-1 > 3:
            end = start+3+1
        for content in lines[start+1:end]:
            scores += float(content.split()[1].strip())
    return scores/(end-start-1)