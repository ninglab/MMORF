from ..utils.logger import Logger
from ..prompts import pruning as pruning_prompts
import os
from .base_agent import BaseAgent
from ..utils.route_review import format_route

with open(os.path.dirname(__file__) + "/../toolsets/pruning.py", "r") as f:
    toolset = f.read()

class Pruning(BaseAgent):
    include_history = False
    def toolset(self):
        return toolset

    def system_prompt(self):
        return pruning_prompts.system_template
    
    def initial_prompt(self, memory):
        return pruning_prompts.initial_prompt(memory)
    
    def continuation_prompt(self, memory):
        return pruning_prompts.continuation_prompt(memory)

    def setup_memory(self, product, routes, step_limit, context=None, reset=False, search=None, **kwargs):
        if isinstance(routes, list):
            routes_dict = {}
            for idx, route in enumerate(routes):
                routes_dict[f"{idx}"] = format_route(route, product)
            routes = routes_dict
        if reset or self.memory is None:
            self.memory = {
                "steps_remaining": step_limit,
                "product": product,
                "context": context,
                "thoughts": [],
                "actions": [],
                "results": [],
                "routes": routes if routes else {},
                "restrictions": {
                    "molecules": [],
                    "specific_reactions": [],
                    "reaction_templates": [],
                    "max_depth": -1
                },
                "value_function": search.vfm.value_fun if search is not None else "Synth()",
                "finished": False,
                "error": False
            }
            if search is not None:
                self.memory["search"] = search
                self.memory["restrictions"] = search.restrictions
            self.messages = [{"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": self.initial_prompt(self.memory)}]
            self.log({"role": "info", "content": "Reset memory."})