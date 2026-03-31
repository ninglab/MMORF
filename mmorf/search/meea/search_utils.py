import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from .meea_value_fn import ValueEnsemble
from .expansion_model import MLPModel
import os

def prepare_value(model_f, device):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

def prepare_expand(model_path, device):
    mlp_rules_path = os.path.dirname(__file__) + "/../../saved_model/template_rules.dat"
    one_step = MLPModel(model_path, mlp_rules_path, device=device)
    return one_step

def prepare_starting_molecules(restrictions=None):
    origin_dict_path = os.path.dirname(__file__) + "/../../prepare_data/origin_dict.csv"
    starting_mols = set(list(pd.read_csv(origin_dict_path)['mol']))
    if restrictions is not None:
        res = set(restrictions.split('.'))
        starting_mols = starting_mols - res
    return starting_mols

def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        print("Invalid SMILES: ", s)
        mol = Chem.MolFromSmiles("")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps

from mmorf.utils.feasibility import fs_feasibility
def is_feasible(reaction_smiles):
    return fs_feasibility(reaction_smiles)