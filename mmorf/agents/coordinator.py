from ..utils.logger import Logger
from ..prompts import coordinator as coordinator_prompts
import os, json
from .base_agent import BaseAgent
from ..utils.route_review import format_route
from typing import List, Union, Dict, Any
from ..search.meea.graph import Node

with open(os.path.dirname(__file__) + "/../toolsets/coordinator.py", "r") as f:
    toolset = f.read()

class Coordinator(BaseAgent):
    include_history = False
    def toolset(self):
        return toolset

    def system_prompt(self):
        return coordinator_prompts.system_template
    
    def initial_prompt(self, memory):
        return coordinator_prompts.initial_prompt(memory)

    def continuation_prompt(self, memory):
        return coordinator_prompts.continuation_prompt(memory)

    def setup_memory(self, product, routes, step_limit, context, value=None, v_synth=None, nodes=None, reset=False, search=None, **kwargs):
        if routes is None:
            routes = []
        if value is None:
            value = [None] * len(routes)
        if v_synth is None:
            v_synth = [None] * len(routes)
        if nodes is None:
            nodes = [None] * len(routes)
        if isinstance(routes, list):
            routes_dict = {}
            for idx, (route, v, vs, n) in enumerate(zip(routes, value, v_synth, nodes)):
                routes_dict[f"{idx}"] = format_route(route, product, node=n, value=v, v_synth=vs)
            routes = routes_dict
        if reset or self.memory is None:
            self.memory = {
                "steps_remaining": step_limit,
                "product": product,
                "context": context,
                "thoughts": [],
                "actions": [],
                "results": [],
                "continue_expand": True,
                "continue_iterations": 0,
                "routes": routes if routes else {},
                "restrictions": {
                    "molecules": [],
                    "specific_reactions": [],
                    "reaction_templates": [],
                    "depth_limit": -1
                },
                "value_function": "Synth()",
                "finished": False,
                "llm": self.llm,
                "error": False
            }
            if search is not None:
                self.memory["search"] = search
                self.memory["restrictions"] = search.restrictions
            self.messages = [{"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": self.initial_prompt(self.memory)}]
            self.log({"role": "info", "content": "Reset memory."})

    @staticmethod
    def create_callback(llm):
        llm.setup_model()
        def callback(value_fn_scores: List[float], original_scores : List[float], nodes : List[Any], search : Any, context: str) -> bool:
            """
            A callback function to be used during MEEA search to prune routes based on LLM evaluations.
            """
            cagent = Coordinator(llm=llm)
            cagent.log({"role": "info", "content": f"Simulation complete.  Current value function: {str(search.vfm.value_fun)}\nCurrent restrictions: {json.dumps(search.restrictions, indent=1)}"})
            if search.continue_iterations > 0:
                # skip LLM evaluation and continue expanding using the previous policy
                search.continue_iterations -= 1
                cagent.log({"role": "info", "content": f"Skipping LLM Coordinator, continuing previous expansion policy. {search.continue_iterations} skip iterations remaining."})
                return True
            # Otherwise, invoke the LLM-based coordinator to guide the search
            mcts_a = search
            if not hasattr(search, "coordinator_memory"):
                search.coordinator_memory = []
            cagent.setup_memory(
                product=search.target_mol,
                routes=[node.partial_route() for node in nodes],
                step_limit=1,
                context=context,
                value=value_fn_scores,
                v_synth=original_scores,
                search=search,
                reset=True,
                nodes=[node for node in nodes]
            )
            cagent.memory["search"].next_node = None
            cagent.messages = [{"role": "system", "content": cagent.system_prompt()},
                                {"role": "user", "content": cagent.initial_prompt(cagent.memory)}]
            cagent.loop(product=search.target_mol, routes=[node.partial_route() for node in nodes], step_limit=1, context=context, reset=False)
            search.coordinator_memory.append(cagent.memory["actions"][-1])
            cagent.memory["search"].continue_iterations = cagent.memory["continue_iterations"]
            if isinstance(cagent.memory["continue_expand"], str):
                cagent.memory["search"].next_node = cagent.memory["routes"][cagent.memory["continue_expand"]]["node"]
            elif isinstance(cagent.memory["continue_expand"], Node):
                cagent.memory["search"].next_node = cagent.memory["continue_expand"]
            return bool(cagent.memory["continue_expand"])
        return callback