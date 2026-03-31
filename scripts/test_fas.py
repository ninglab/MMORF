from mmorf.search.meea import MEEA
from mmorf.search.meea.search import MCTS_A
from mmorf.search.meea.graph import Node
from mmorf.llms.claude4 import Claude
from mmorf.llms.local import Qwen30BInstruct
from mmorf.agents.coordinator import Coordinator
from mmorf.llms.local import Qwen30BThinking, MistralNemo12B
from mmorf.agents.value_function_modification import ValueFunctionModification
from mmorf.agents.pruning import Pruning
from mmorf.agents.accept_reject import AcceptReject
from dotenv import load_dotenv
from typing import List
import numpy as np
import warnings
import torch, argparse, numpy
import numpy.core
import os

import importlib
import argparse

import mmorf.utils.route_review
import mmorf.agents.pruning
load_dotenv(".env")

llm = Claude()
if os.environ.get("REPLANNER_LLM", "Claude") == "Qwen30B":
    from mmorf.llms.local import Qwen30BInstruct
    llm = Qwen30BInstruct()
    print("Using Qwen30BInstruct LLM")
elif os.environ.get("REPLANNER_LLM", "Claude") == "Mistral":
    llm = MistralNemo12B()
    print("Using Mistral Nemo 12B")
llm.setup_model()

ap = argparse.ArgumentParser()
ap.add_argument("--fs", "--feasibility_check", action="store_true", help="Whether to perform feasibility checking of reactions.")
ap.add_argument("--exp", "--expand_intermediate", action="store_true", help="Whether to enable further expansion.")
ap.add_argument("--auto", type=int, default=10, help="Number of iterations to run automatically before engaging the agent.")
ap.add_argument("--auto_exp", "--auto_further_expand", action="store_true", help="Whether to automatically further expand nodes on first expansion")
ap.add_argument("--device", type=str, default="cpu", help="Device to run the search on.")
ap.add_argument("--output", type=str, default=f"""output_{os.environ.get("SLURM_ARRAY_TASK_ID","None")}.json""", help="Output file to save the results.")
ap.add_argument("--context", type=str, default=None, help="Constraints or context for the retrosynthesis planning.")
ap.add_argument("product", type=str, help="SMILES of the target product molecule.")
args = ap.parse_args()
os.environ["ENABLE_FURTHER_EXPANSION"] = "1" if args.exp else "0"

cache_dir = os.environ.get("REPLANNER_CACHE_DIR", None)
if cache_dir is not None:
    os.makedirs(cache_dir, exist_ok=True)

meea_search = MEEA(device=args.device, feasible=args.fs, output=cache_dir, enable_further_expansion=args.exp)
# meea_search.interrupt_mgr.register("simulation_complete", Coordinator.create_callback(llm))

def callback(node, route, search) -> bool:
    print("Starting Restrictions:", search.restrictions)
    arcb = AcceptReject.create_callback(llm)
    accepted = arcb(node, route, search)
    if accepted:
        print("Route accepted by Agent:")
        print(route)
        return True
    print("Route rejected by Agent.")
    # agent = ValueFunctionModification(llm=llm, search=search)
    # agent.loop(search.target_mol, search.rejected_routes[-3:], context=search.feedback, steplimit=3)
    agent2 = Pruning(llm=llm)
    context = search.task_context if search.task_context is not None else ""
    if search.feedback:
        context += "\n" + search.feedback
    if context.strip() == "":
        context = None
    routes = [r["reactions"] for r in search.rejected_routes[-5:]]
    agent2.setup_memory(search.target_mol, routes, step_limit=3, context=context, search=search, reset=True)
    agent2.loop(search.target_mol, routes, context=context, step_limit=3, reset=False)
    search.restrictions = agent2.memory["restrictions"]
    search.revise_restrictions()
    print("Ending Restrictions:", search.restrictions)
    return False

meea_search.interrupt_mgr.register("route_found", callback)
if args.exp:
    meea_search.interrupt_mgr.register("expansion_coordinator", Expansion.create_callback(llm))
meea_search.interrupt_mgr.register("simulation_complete", lambda *args, **kwargs: True)
smiles = args.product
results = meea_search(smiles, context=args.context, k=100, iteration_limit=500, start_iter=args.auto, auto_further_expand=args.auto_exp)

print("Final Results:")
print(results)

with open(args.output, "w") as f:
    import json
    json.dump(results, f)