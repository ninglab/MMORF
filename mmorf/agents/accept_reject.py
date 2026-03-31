from ..utils.logger import Logger
from ..prompts import accept_reject as accept_reject_prompts
import os
from .base_agent import BaseAgent
from ..utils.route_review import format_route
from typing import Any, List, Dict

with open(os.path.dirname(__file__) + "/../toolsets/accept_reject.py", "r") as f:
    toolset = f.read()

class AcceptReject(BaseAgent):
    include_history = False
    def toolset(self):
        return toolset

    def system_prompt(self):
        return accept_reject_prompts.system_template
    
    def initial_prompt(self, memory):
        return accept_reject_prompts.initial_prompt(memory)

    def continuation_prompt(self, memory):
        return accept_reject_prompts.continuation_prompt(memory)

    def setup_memory(self, product, routes, context, reset=False, search=None, **kwargs):
        if routes is None:
            routes = []
        if isinstance(routes, list):
            routes_dict = {}
            for idx, route in enumerate(routes):
                print(route)
                routes_dict[f"{idx}"] = format_route(route, product)
                del routes_dict[f"{idx}"]['value']
                del routes_dict[f"{idx}"]['value_synth_only']
            routes = routes_dict
        if reset or self.memory is None:
            self.memory = {
                "product": product,
                "iterations": search.times - search.iterations,
                "route": list(routes.values())[0],
                "context": context,
                "thoughts": [],
                "actions": [],
                "results": [],
                "accepted": False,
                "steps_remaining": 1,
                "rejected": False,
                "feedback": "",
                "finished": False,
                "llm": self.llm,
                "error": False
            }
            if search is not None:
                self.memory["search"] = search
            self.messages = [{"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": self.initial_prompt(self.memory)}]
            self.log({"role": "info", "content": "Reset memory."})

    @staticmethod
    def create_callback(llm):
        llm.setup_model()
        def callback(node : Any, route : List[str], search : Any) -> bool:
            """
            A callback function to be used during MEEA search to accept or reject proposed routes based on LLM evaluations.
            """
            aragent = AcceptReject(llm=llm)
            aragent.setup_memory(
                product=search.target_mol,
                routes=[route],
                context=search.task_context,
                search=search,
                reset=True,
            )
            aragent.loop(product=search.target_mol, routes=[route], step_limit=1, context=search.task_context, reset=False)
            if aragent.memory["accepted"]:
                search.route = aragent.memory["route"]
                return True
            return False
        return callback
    
class AcceptRejectChange(AcceptReject):
    def system_prompt(self):
        return accept_reject_prompts.system_template_change
    
    @staticmethod
    def create_callback(llm):
        llm.setup_model()
        def callback(node : Any, route : List[str], search : Any) -> bool:
            """
            A callback function to be used during MEEA search to accept or reject proposed routes based on LLM evaluations.
            """
            aragent = AcceptRejectChange(llm=llm)
            aragent.setup_memory(
                product=search.target_mol,
                routes=[route],
                context=search.task_context,
                search=search,
                reset=True,
            )
            aragent.loop(product=search.target_mol, routes=[route], step_limit=1, context=search.task_context, reset=False)
            if aragent.memory["accepted"]:
                search.route = aragent.memory["route"]
                return True
            return False
        return callback