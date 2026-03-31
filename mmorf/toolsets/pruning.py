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

def get_restricted_routes(mem):
    restricted_routes = {}
    for id, route in mem["routes"].items():
        g = create_route_graph(route["reactions"], canonical=True)
        if mem["restrictions"]["max_depth"] != -1:
            # prune reactions beyond the depth limit
            for n in list(g.nodes):
                if ">>" in n:
                    depth = get_reaction_depth(g, n)
                    if depth > mem["restrictions"]["max_depth"]:
                        # collect all downstream nodes to remove
                        descendants = nx.descendants(g, n)
                        for d in descendants:
                            if d in g.nodes:
                                g.remove_node(d)
                        g.remove_node(n)
        for m in mem["restrictions"]["molecules"]:
            # prune any reaction that uses this molecule as a reactant
            for r in list(g.nodes):
                if ">>" in r:
                    reactants = r.split(">>")[0].split(".")
                    if m in reactants:
                        # collect all downstream nodes to remove
                        descendants = nx.descendants(g, r)
                        for d in descendants:
                            if d in g.nodes:
                                g.remove_node(d)
                        g.remove_node(r)
        for rxn in mem["restrictions"]["specific_reactions"]:
            # prune any reaction that matches this reaction exactly
            for r in list(g.nodes):
                if ">>" in r:
                    if r.split(">>")[1].strip() == rxn.split(">>")[1].strip(): # products match
                        reactants = canonicalize(r.split(">>")[0]).split(".")
                        rxn_reactants = canonicalize(rxn.split(">>")[0]).split(".")
                        if set(reactants) == set(rxn_reactants): # reactants match
                            # collect all downstream nodes to remove
                            descendants = list(nx.descendants(g, r))
                            for d in descendants:
                                if d in g.nodes:
                                    g.remove_node(d)
                            g.remove_node(r)
        for smarts in mem["restrictions"]["reaction_templates"]:
            patt = AllChem.ReactionFromSmarts(smarts)
            patt.Initialize()
            nodelist = list(g.nodes)
            for r in nodelist:
                if ">>" in r:
                    try:
                        rxn = reaction_from_smiles(r.split(">>")[0], r.split(">>")[1])
                        rxn.Initialize()
                        # see if rxn matches patt
                        if rdChemReactions.HasReactionSubstructMatch(rxn, patt):
                            # collect all downstream nodes to remove
                            descendants = list(nx.descendants(g, r))
                            for d in descendants:
                                if d in g.nodes:
                                    g.remove_node(d)
                            g.remove_node(r)
                    except Exception as e:
                        continue
        reactions = [n for n in g.nodes if ">>" in n]
        restricted_routes[id] = format_route(reactions, route["product"]) if len(reactions) > 0 else {"product": route["product"], "reactions": [], "length": 0}
    return restricted_routes


def RestrictMolecules(*smiles_list: str):
    """
    Restrict certain molecules from being used in future synthesis planning.
    
    Args:
        smiles_list (str): One or more SMILES strings to restrict.
        
    Returns:
        str: The action to restrict the specified SMILES.
    """
    print("DEBUGGING>>> RestrictMolecules called with:", smiles_list)
    for smiles in smiles_list:
        if canonicalize(smiles) == "Invalid SMILES.":
            return f"Invalid SMILES: {smiles}. Please provide valid SMILES strings.", True
    for smiles in smiles_list:
        print(smiles)
        if smiles not in memory["restrictions"]["molecules"]:
            smiles = canonicalize(smiles)
            memory["restrictions"]["molecules"].append(smiles)
            print("DEBUGGING>>> Added SMILES restriction:", smiles, memory["restrictions"]["molecules"])
    print("DEBUGGING>>> Final SMILES restrictions before updating search:", memory["restrictions"])
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    print("DEBUGGING>>> RestrictMolecules updated restrictions:", memory["restrictions"])
    return f"""Updated restricted SMILES list: {', '.join(memory['restrictions']['molecules'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False
    
def RestrictSpecificReactions(*reaction_list: str):
    """
    Restrict certain reactions from being used in future synthesis planning.
    
    Args:
        reaction_list (str): One or more reaction SMILES strings to restrict.
        
    Returns:
        str: The action to restrict the specified reactions.
    """
    for reaction in reaction_list:
        try:
            rxn = reaction_from_smiles(reaction.split(">>")[0], reaction.split(">>")[1])
            rxn.Initialize()
        except Exception as e:
            return f"Invalid reaction SMILES: {reaction}. Please provide valid reaction SMILES strings.", True
    for reaction in reaction_list:
        if reaction not in memory["restrictions"]["specific_reactions"]:
            memory["restrictions"]["specific_reactions"].append(reaction)
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Updated restricted reactions list: {', '.join(memory['restrictions']['specific_reactions'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def RestrictReactionTemplates(*smarts_list: str):
    """
    Restrict certain reaction templates from being used in future synthesis planning.
    
    Args:
        smarts_list (str): One or more reaction SMARTS strings to restrict.
        
    Returns:
        str: The action to restrict the specified reaction templates.
    """
    for smarts in smarts_list:
        try:
            patt = AllChem.ReactionFromSmarts(smarts)
            patt.Initialize()
        except Exception as e:
            return f"Invalid reaction SMARTS: {smarts}. Please provide valid reaction SMARTS strings.", True
    for smarts in smarts_list:
        if smarts not in memory["restrictions"]["reaction_templates"]:
            memory["restrictions"]["reaction_templates"].append(smarts)
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Updated restricted reaction templates list: {', '.join(memory['restrictions']['reaction_templates'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def DepthLimit(depth):
    """
    Set a maximum depth for synthesis routes.
    
    Args:
        depth (int): The maximum depth for synthesis routes. Use -1 to remove any depth limit.
        
    Returns:
        str: The action to set the depth limit.
    """
    if not isinstance(depth, int) or depth < -1 or depth == 0:
        return f"Invalid depth limit: {depth}. Please provide a positive integer or -1 to remove the limit.", True
    memory["restrictions"]["max_depth"] = depth
    if depth == -1:
        return f"Removed depth limit.  "+json.dumps(get_restricted_routes(memory), indent=1), False
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Set depth limit to {depth}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def UnrestrictMolecules(*smiles_list: str):
    """
    Remove restrictions on certain molecules.
    
    Args:
        smiles_list (str): One or more SMILES strings to unrestrict.
        
    Returns:
        str: The action to unrestrict the specified SMILES.
    """
    for smiles in smiles_list:
        smiles = canonicalize(smiles)
        if smiles in memory["restrictions"]["molecules"]:
            memory["restrictions"]["molecules"].remove(smiles)
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Updated restricted SMILES list: {', '.join(memory['restrictions']['molecules'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def UnrestrictSpecificReaction(*reaction_list: str):
    """
    Remove restrictions on certain reactions.
    
    Args:
        reaction_list (str): One or more reaction SMILES strings to unrestrict.
        
    Returns:
        str: The action to unrestrict the specified reactions.
    """
    for reaction in reaction_list:
        if reaction in memory["restrictions"]["specific_reactions"]:
            memory["restrictions"]["specific_reactions"].remove(reaction)
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Updated restricted reactions list: {', '.join(memory['restrictions']['specific_reactions'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def UnrestrictReactionTemplate(*smarts_list: str):
    """
    Remove restrictions on certain reaction templates.
    
    Args:
        smarts_list (str): One or more reaction SMARTS strings to unrestrict.
        
    Returns:
        str: The action to unrestrict the specified reaction templates.
    """
    for smarts in smarts_list:
        if smarts in memory["restrictions"]["reaction_templates"]:
            memory["restrictions"]["reaction_templates"].remove(smarts)
    memory["search"].restrictions = memory["restrictions"]
    memory["search"].revise_restrictions()
    return f"""Updated restricted reaction templates list: {', '.join(memory['restrictions']['reaction_templates'])}.\nUpdated routes: """+json.dumps(get_restricted_routes(memory), indent=1), False

def Finalize():
    """
    Finalize the pruning process and return the pruned routes.
    
    Returns:
        str: The pruned routes.
    """
    print("DEBUGGING>>> Finalize called. Restrictions:", memory["search"].restrictions, memory["restrictions"])
    memory["finished"] = True
    # memory["search"].restrictions = memory["restrictions"]
    # memory["search"].revise_restrictions()
    return f"""Pruning finalized. Making restrictions permanent: {json.dumps(memory['restrictions'], indent=1)}""", False