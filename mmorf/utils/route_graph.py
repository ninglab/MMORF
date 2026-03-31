import networkx as nx
import os, json
from .canonicalize import canonicalize

def create_route_graph(route, canonical=False):
    """
    Create a directed graph from a synthesis route.

    Args:
        route (list): A list of strings representing the synthesis route,
                      where each string is a reaction in the format "reactant.reactant...>>product".

    Returns:
        nx.DiGraph: A directed graph representing the synthesis route.
    """
    G = nx.DiGraph()
    
    for reaction in route:
        parts = reaction.split(">")
        if len(parts) < 2:
            continue  # Skip invalid reactions
        
        reactants = parts[0].split(".")
        product = parts[-1].strip()
        if canonical:
            reactants = [canonicalize(r.strip()) for r in reactants]
            product = canonicalize(product.strip())
        else:
            reactants = [r.strip() for r in reactants]
            product = product.strip()
        
        for reactant in reactants:
            G.add_edge(reactant.strip(), reaction.strip())
        G.add_edge(reaction.strip(), product.strip())
    return G

def get_reaction_depth(route, reaction_or_molecule):
    route_sanitized = [r.strip() for r in route if r is not None and r.strip()]
    route = route_sanitized
    reaction_or_molecule = reaction_or_molecule.strip()
    if os.environ.get("REPLANNER_DEBUG", "0") == "1":
        print(f"Creating route graph for: {json.dumps(route, indent=2)}")
    route_graph = create_route_graph(route)
    if os.environ.get("REPLANNER_DEBUG", "0") == "1":
        print(f"Route graph created with {len(route_graph.nodes)} nodes and {len(route_graph.edges)} edges.")
        if not nx.is_directed_acyclic_graph(route_graph):
            print("Warning: The route graph is not a directed acyclic graph (DAG).  Returning -1 for depth")
            return -1
        print(f"Connected components: {list(nx.connected_components(route_graph.to_undirected()))}")
    if reaction_or_molecule not in route_graph.nodes:
        return -1
    root = [n for n, d in route_graph.out_degree() if d == 0][0]
    # count only reaction nodes towards depth, not molecule nodes
    depth = 0
    path = nx.shortest_path(route_graph.reverse(), source=root, target=reaction_or_molecule)
    for node in path:
        if ">>" in node:
            depth += 1
    return depth

    