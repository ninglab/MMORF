import os, json
from .run_once import run_once
from rdkit import Chem

@run_once
def load_purch_list():
    list_of_purch = []
    with open(os.path.dirname(__file__) + "/../prepare_data/origin_dict.csv", "r") as f:
        for line in f:
            line = line.strip().split(",")[1]
            list_of_purch.append(line)
    return list_of_purch

@run_once
def load_purch_cost():
    if not os.path.exists(os.path.dirname(__file__) + "/../prepare_data/purch_cost.json"):
        print("Warning: purch_cost.json not found.")
        return {}
    with open(os.path.dirname(__file__) + "/../prepare_data/purch_cost.json", "r") as f:
        return json.load(f)

purch = load_purch_list()
purch_cost = load_purch_cost()
