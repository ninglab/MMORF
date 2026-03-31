import time
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
from mmorf.agents.pruning import Pruning as PruningAgent
from mmorf.agents.value_function_modification import ValueFunctionModification as ValueFnModAgent
from mmorf.utils.route_review import format_route
import json

def AcceptProposed(reason):
    """Accept the current route

    Args:
        instructions (str): Instructions for pruning routes.

    Returns:
        str: Result of the pruning operation.
    """
    # Placeholder implementation
    memory["accepted"] = True
    memory["rejected"] = False
    memory["finished"] = True
    memory["route"]["feedback"] = "Route accepted due to the following reason: " + reason
    return "Route accepted.", False

def AcceptPrevious(id, reason):
    """Accept the current route

    Args:
        instructions (str): Instructions for pruning routes.

    Returns:
        str: Result of the pruning operation.
    """
    # Placeholder implementation
    memory["accepted"] = True
    memory["rejected"] = False
    memory["finished"] = True
    memory["route"]["feedback"] = "Rejected to accept a previous route for this reason: "+ reason
    memory["search"].rejected_routes.append(memory["route"])
    memory["route"] = memory["search"].rejected_routes[id]
    memory["route"]["feedback"] = "\nRoute later accepted due to the following reason:\n"+ reason
    return "Previous route accepted.", False

def AcceptModified(route, reasoning):
    """Accept the current route with modifications

    Args:
        instructions (str): Instructions for pruning routes.

    Returns:
        str: Result of the pruning operation.
    """
    # Placeholder implementation
    memory["search"].rejected_routes.append(memory["route"])
    memory["route"]["feedback"] = "Rejected as-is, but accepted a modified version with the following reasoning: "+ reasoning
    memory["route"] = format_route(route, memory["product"])
    memory["route"]["feedback"] += "\nThis route is a modification of other routes, and was accepted due to the following reasoning:\n"+ reasoning
    memory["accepted"] = True
    memory["rejected"] = False
    memory["finished"] = True
    return "Modified route accepted.", False

def Reject(feedback):
    """Reject the current route with feedback.

    Args:
        feedback (str): Feedback on why the route is not acceptable.
    Returns:
        str: Result of the value function modification.
    """
    memory["feedback"] = feedback
    memory["accepted"] = False
    memory["rejected"] = True
    memory["finished"] = True
    memory["search"].feedback = feedback
    memory["route"]["feedback"] = "Route rejected with feedback: " + feedback
    memory["search"].rejected_routes.append(memory["route"])
    return "Route rejected with feedback.", False