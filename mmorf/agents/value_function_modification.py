from ..utils.logger import Logger
from ..prompts import value_function_modification as prompts
from ..utils.route_review import format_route
import os
from .base_agent import BaseAgent

with open(os.path.dirname(__file__) + "/../toolsets/value_function_modification.py", "r") as f:
    toolset = f.read()

class ValueFunctionModification(BaseAgent):
    include_history = False
    def __init__(self, llm, search):
        super().__init__(llm)
        self.search = search

    def toolset(self):
        return toolset

    def system_prompt(self):
        return prompts.system_template
    
    def initial_prompt(self, memory):
        return prompts.initial_prompt(memory)
    
    def continuation_prompt(self, memory):
        return prompts.continuation_prompt(memory)

    def setup_memory(self, product, routes, step_limit, context, nodes=None, value=None, v_synth=None, reset=False):
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
                "search": self.search,
                "thoughts": [],
                "actions": [],
                "results": [],
                "routes": routes if routes else {},
                "value_function": self.search.vfm.value_fun,
                "finished": False,
                "error": False
            }
            self.messages = [{"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": self.initial_prompt(self.memory)}]
            self.log({"role": "info", "content": "Reset memory."})