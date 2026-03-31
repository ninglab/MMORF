import time
from mmorf.utils.canonicalize import canonicalize
from mmorf.utils.route_graph import create_route_graph, get_reaction_depth
from mmorf.utils.template_matching import reaction_from_smiles
from mmorf.utils.route_review import format_route
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import pandas as pd
import numpy as np
import networkx as nx
import torch, argparse, numpy
torch.serialization.add_safe_globals([argparse.Namespace])
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([numpy.ndarray])
torch.serialization.add_safe_globals([numpy.dtype])
torch.serialization.add_safe_globals([numpy.dtypes.Float64DType])
import json

def SetValueFunction(value_fn : str):
    """
    Set the value function.
        
    Returns:
        str: A message indicating that the value function has been set.
    """
    def Synth():
        return 1.0
    def BBPrice():
        return 1.0
    def Depth():
        return 1.0
    def GHS(*hazard_codes):
        assert len(hazard_codes) > 0, "At least one hazard code must be provided."
        return 1.0
    def FastCarc():
        return 1.0
    def SlowCarc():
        return 1.0
    def MaxSim(smiles):
        return 1.0
    def MinSim(smiles):
        return 1.0
    def Pyro():
        return 1.0
    # Fill in additional dummy functions from prompt in ../prompts/value_function_modification.py as needed.
    try: # initial quick test to see if the value function is valid.
        x = eval(value_fn)
        assert isinstance(x, float) or isinstance(x, int), "Value function must evaluate to a numeric score."
    except Exception as e:
        return f"Error setting value function: {e}", True
    old_value_fn = memory["value_function"]
    memory["value_function"] = value_fn
    try:
        memory["search"].vfm.set_value_function(value_fn)
        for route_id, route in memory["routes"].items():
            route_value = memory["search"].vfm.all_evaluate(v_orig=route["value_synth_only"],
                                                            node=route["node"],
                                                            output=memory["search"].output)
            route["value"] = route_value
    except Exception as e: # Undo the change if evaluation fails.
        memory["search"].vfm.set_value_function(old_value_fn)
        memory["value_function"] = old_value_fn
        for route_id, route in memory["routes"].items():
            route_value = memory["search"].vfm.all_evaluate(v_orig=route["value_synth_only"],
                                                            node=route["node"],
                                                            output=memory["search"].output)
            route["value"] = route_value
        return f"Error evaluating value function on existing routes: {e}", True
    return f"""Value function set to \"{value_fn}\".""", False

def Finalize():
    """
    Finalize the value function modification process and return the pruned routes.
    
    Returns:
        str: The pruned routes.
    """
    memory["finished"] = True
    return f"""Value Function Modification finalized: \"{memory["value_function"]}\"""", False