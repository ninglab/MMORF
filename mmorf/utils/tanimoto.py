import rdkit
import os
import json, pickle
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
import time

CACHE_DIR = os.environ.get("mmorf_CACHE_DIR", os.path.dirname(__file__))
fpsim_cache = {}
if not os.environ.get("mmorf_NO_CACHE", False):
    if os.path.exists(CACHE_DIR + "/fpsim_cache.pkl"):
        with open(CACHE_DIR + "/fpsim_cache.pkl", "rb") as f:
            retries = 10
            while len(fpsim_cache) == 0 and retries > 0:
                try:
                    fpsim_cache = pickle.load(f)
                except Exception:
                    fpsim_cache = {}
                    time.sleep(1/retries)
                    retries -= 1

def tanimoto_similarity(smi1, smi2, sanitize=True):
    """Calculate the Tanimoto similarity between two SMILES strings."""
    if (smi1, smi2) in fpsim_cache:
        return fpsim_cache[(smi1, smi2)]
    elif (smi2, smi1) in fpsim_cache:
        return fpsim_cache[(smi2, smi1)]
    else:
        mol1 = rdkit.Chem.MolFromSmiles(smi1, sanitize=sanitize)
        mol2 = rdkit.Chem.MolFromSmiles(smi2, sanitize=sanitize)
        if mol1 is not None and mol2 is not None:
            if len(mol1.GetBonds()) == len(mol2.GetBonds()) \
              and len(mol2.GetBonds()) == 0 \
              and smi1 != smi2:
                fpsim_cache[(smi1, smi2)] = 0.0
                return fpsim_cache[(smi1, smi2)]
        fps = [rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2) for mol in [mol1, mol2]]
        fpsim_cache[(smi1, smi2)] = rdkit.DataStructs.FingerprintSimilarity(fps[0], fps[1])
        return fpsim_cache[(smi1, smi2)]
    
def save_fpsim_cache():
    if not os.environ.get("mmorf_NO_CACHE", False):
        with open(CACHE_DIR + "/fpsim_cache.pkl", "wb") as f:
            f.write(pickle.dumps(fpsim_cache))