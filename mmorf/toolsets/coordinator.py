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
import json

def Pruning(instructions):
    """Prune routes based on the provided instructions.

    Args:
        instructions (str): Instructions for pruning routes.

    Returns:
        str: Result of the pruning operation.
    """
    # Placeholder implementation
    agent = PruningAgent(memory["llm"])
    print("Runing pruning agent with instructions:", instructions)
    agent.setup_memory(memory["product"], memory["routes"], step_limit=3, context=instructions, search=memory["search"], reset=True)
    agent.memory["restrictions"] = memory["restrictions"] = memory["search"].restrictions
    success = agent.loop(memory["product"], memory["routes"], instructions, step_limit=3, reset=False)
    print("Pruning agent finished with success:", success)
    if success:
        memory["restrictions"] = agent.memory["restrictions"]
        memory["search"].restrictions = agent.memory["restrictions"]
        memory["continue_expand"] = False
        memory["continue_iterations"] = 1 # Do 1 expansion after this action
        memory["finished"] = True
    else:
        memory["error"] = True
        return "Pruning agent failed to finish successfully", True
    return "Pruning completed based on instructions.", False

def ValueFn(instructions):
    """Modify the value function based on the provided instructions.

    Args:
        instructions (str): Instructions for modifying the value function.

    Returns:
        str: Result of the value function modification.
    """
    agent = ValueFnModAgent(memory["llm"], memory["search"])
    agent.setup_memory(memory["product"], memory["routes"], step_limit=3, context=instructions, reset=True)
    agent.memory["value_function"] = memory["value_function"]
    success = agent.loop(memory["product"], memory["routes"], instructions, step_limit=3, reset=False)
    if success and agent.memory["finished"]:
        memory["value_function"] = agent.memory["value_function"]
    else:
        memory["error"] = True
        return "Value function modification agent failed to finish successfully.", True
    memory["continue_expand"] = False
    memory["continue_iterations"] = 1 # Do 1 expansion after this action
    memory["finished"] = True
    return "Value function modified based on instructions.", False

def ExpandDefault(N):
    """Expand the route with the highest value and perform N more iterations.

    Args:
        N (int): Number of additional iterations to perform.

    Returns:
        str: Result of the expansion operation.
    """
    memory["continue_expand"] = True
    memory["continue_iterations"] = N
    memory["finished"] = True
    return f"Expanded highest value route and performed {N} additional iterations.", False

def Expand(route_id):
    """Expand the specified route.

    Args:
        route_id (str): Identifier of the route to expand.

    Returns:
        str: Result of the expansion operation.
    """
    memory["continue_expand"] = memory["routes"][str(route_id)]["node"]
    memory["continue_iterations"] = 0
    memory["finished"] = True
    return f"Expanded route with ID {route_id}.", False

def ExpandIntermediate(route_id, smiles):
    """Expand the specified route starting from the specified intermediate molecule.

    Args:
        route_id (str): Identifier of the route to expand.
        smiles (str): SMILES string of the intermediate molecule to start from.

    Returns:
        str: Result of the expansion operation.
    """
    route = memory["routes"][str(route_id)]
    target_smiles = smiles
    target_node = None
    # Traverse the route's tree backwards to find the node with the intermediate at its state[0]
    nodes_to_visit = [route["node"]]
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if current_node.state[0] == target_smiles:
            target_node = current_node
            break
        if current_node.parent is not None:
            nodes_to_visit.append(current_node.parent)
    if target_node is None:
        memory["error"] = True
        return f"Intermediate molecule {smiles} not found in route {route_id}, make sure this intermediate has already been explored.", True
    memory["continue_expand"] = target_node
    memory["continue_iterations"] = 0
    memory["finished"] = True
    return f"Expanding route {route_id} starting from intermediate {smiles}.", False