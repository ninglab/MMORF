import os
from .run_once import run_once
from rdkit import Chem

@run_once
def load_pyro_list():
    list_of_pyro = []
    with open(os.path.dirname(__file__) + "/../value_functions/pyrophoricity.csv", "r") as f:
        for line in f:
            line = line.strip().split(",")[1]
            if Chem.MolFromSmiles(line) is not None:
                 list_of_pyro.append(line)
    return list_of_pyro 

pyro = load_pyro_list()
