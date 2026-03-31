from mmorf.utils.canonicalize import canonicalize
from mmorf.utils.route_graph import create_route_graph, get_reaction_depth
from mmorf.utils.admet_ai import carc_cache, get_admet_model, save_carc_cache
from mmorf.utils.bbsa import bbsa_alerts
from mmorf.utils.tanimoto import tanimoto_similarity, save_fpsim_cache
from mmorf.utils.pubchem import get_ghs_from_pubchem
from mmorf.utils.feasibility import fs_feasibility, llm_feasibility, rfm_feasibility
from mmorf.utils.template_matching import reaction_from_smiles
# from mmorf.utils.coprinet_apptainer import cost_estimate
from mmorf.utils.pyro_list import pyro
from mmorf.utils.purch_list import purch, purch_cost
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import os
import numpy as np
import pandas as pd

def cost_calculator(rt):
    g = create_route_graph(rt)
    mols = [n for n in g.nodes if g.in_degree(n) == 0 and ">>" not in n]  # Get all unsynthesized molecules (purchasable reactants)
    if len(mols) == 0:
        return "Unknown"
    cost_estimates = [purch_cost.get(m, 0.0) for m in mols]
    costs = []
    for cost in cost_estimates:
        if isinstance(cost, str):
            if cost.lower() == "unknown":
                costs.append(os.environ.get("REPLANNER_DEFAULT_COST", np.exp(4.37)))  # Default cost if unknown
            else:
                costs.append(float(cost))
        else:
            costs.append(cost)
    return sum(costs) if len(costs) > 0 else "Unknown"
def pyro_calculator(rt):
    mols = ".".join(rt).replace(">>", ".").replace("\n","").split(".")
    for m in mols:
        for p in pyro:
            try:
                if tanimoto_similarity(m, p) >= 1.0:
                    return 1.0
            except Exception as e:
                pass # Ignore errors in similarity calculation
    return 0.0
def carc_calculator(rt, fast=False):
    mols = ".".join(rt).replace(">>", ".").replace("\n","").split(".")
    if os.environ.get("FAST_CARCINOGENICITY", "0") == "1" or fast:
        return int(any([bbsa_alerts(m) for m in mols]))
    model = get_admet_model()
    carc_dict = {}
    need_preds = []
    for m in mols:
        if canonicalize(m) is None:
            carc_dict[m] = "Invalid SMILES"
            continue
        if m in carc_cache:
            carc_dict[m] = carc_cache[m]
        elif Chem.MolToSmiles(Chem.MolFromSmiles(m)) in carc_cache:
            carc_dict[m] = carc_cache[Chem.MolToSmiles(Chem.MolFromSmiles(m))]
        else:
            need_preds.append(m)
    if len(need_preds) > 0:
        preds = model.predict(need_preds + ["CCO", "O"]) # adding "CCO" and "O" to ensure the returned value is consistently a DataFrame
        for m in need_preds:
            if m in preds.index:
                carc_dict[m] = preds.loc[m, "Carcinogens_Lagunin"]
                if isinstance(carc_dict[m], pd.Series): # can happen if "CCO" or "O" is in the need_preds
                    carc_dict[m] = carc_dict[m].max()
            else:
                carc_dict[m] = "Unknown"
            carc_cache[m] = carc_dict[m]
        save_carc_cache()
    return max([carc_dict[m] for m in mols if not isinstance(carc_dict[m], str)] + [0.0])  # Return the maximum carcinogenicity score

def get_iupac(smiles):
    # request from cactus server
    import requests
    try:
        smiles = canonicalize(smiles)
        # URLEncode SMILES (e.g. # to %23)
        from urllib.parse import quote
        smiles = quote(smiles)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            result = response.text.strip()
            if len(result) > 5000:
                return "Error in resolving IUPAC name"
            return result
        else:
            return "Could not resolve IUPAC name"
    except Exception as e:
        print(e)
        return "Could not resolve IUPAC name"

def ghs_calculator(rt):
    mols = ".".join(rt).replace(">>", ".").replace("\n","").split(".")
    return list(sorted(set().union(*[set(get_ghs_from_pubchem(m)) for m in mols if canonicalize(m) is not None])))

def format_route(route, smiles, node=None, v_synth=None, value=None, use_iupac=True):
    print(route)
    products = [x.split(">>")[1].strip() for x in route]
    molecules = set().union(*[rxn.replace(">>", ".").split(".") for rxn in route])
    rxndepth_map = {m: min([i for i, rxn in enumerate(route) if m in rxn.replace(">",".").split(".")]) for m in molecules}
    id = {
        "product": smiles,
        "reactions": [
            rxn for rxn in route
        ],
        "length": len(route),
        "value": value if value is not None else (v_synth if v_synth is not None else 0.0),
        "value_synth_only": v_synth if v_synth is not None else 0.0,
        "GHS": ghs_calculator(route),
        "Maximum SlowCarc Score": carc_calculator(route),
        "Maximum FastCarc Score": carc_calculator(route, fast=True),
        "Maximum Pyrophoricity Score": pyro_calculator(route),
        "Total Cost for 1g of each starting material": cost_calculator(route),
        "molecule_details": {
            s.strip(): {
                "IUPAC_name": get_iupac(s) if use_iupac else "Not requested",
                "SMILES_with_explicit_hydrogens": Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(s)), isomericSmiles=True, allHsExplicit=True),
                "pyrophoricity (Similarity-based alert score)": pyro_calculator([s]),
                "carcinogenicity": {
                    "SlowCarc (Risk score, not always accurate)": carc_calculator([s]),
                    "FastCarc (Binary alert indicator, not always accurate)": carc_calculator([s], fast=True),
                },
                "reaction": rxndepth_map[s],
                "is_building_block": s in purch,
                "depth_in_route": rxndepth_map[s],
                "ghs": ghs_calculator([s]),
                "estimated_cost": cost_calculator([f"{s}>>C"]) if s not in products else "intermediate, not purchased"
            } for s in molecules
        }
    }
    if node is not None:
        id["node"] = node
        # id["can_be_expanded"] = not node.is_expanded
    return id