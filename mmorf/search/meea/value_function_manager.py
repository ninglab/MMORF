import re
from .graph import Node
from ...utils.logger import Logger
from mmorf.value_functions import *


class ValueFunctionManager:
    def __init__(self, value_fun: str):
        self.set_value_function(value_fun)

    def set_value_function(self, value_fun: str):
        self.value_fun = value_fun
        self.parsed_function = None
        value_fun_lambda = value_fun.strip()
        Logger.log("DEBUGGING>>> Evaluating value function: "+value_fun.strip())
        self.parsed_function = eval(value_fun_lambda)
        try:
            test_output = self.parsed_function(reaction="C.C>>CC", v_orig=1.0, output="test_output", node=None, is_leaf=True, force_compute=True)
        except Exception as e:
            Logger.log("ERROR>>> Value function evaluation failed during testing with error: "+str(e))
            raise e
        Logger.log("DEBUGGING>>> Parsed function: "+repr(eval(value_fun))) # repr(eval(value_fun))) is correct.

    def evaluate(self, v_orig : float, reaction : str, output : str, node : Node, is_leaf=False, force_compute=False):   
        return self.parsed_function(reaction=reaction, v_orig=v_orig, output=output, node=node, is_leaf=is_leaf, force_compute=force_compute)

    def all_evaluate(self, v_orig : float, node : Node, output : str, force_compute=False):
        return self.evaluate(v_orig, node.reaction, output, node=node, force_compute=force_compute)
        # if node.parent is None:
        #     if node.reaction is None:
        #         return v_orig
        #     else:
        #         return self.evaluate(v_orig, node.reaction, output, node=node, is_leaf=is_leaf, force_compute=force_compute)
        # else:
        #     if node.reaction is None:
        #         return self.all_evaluate(v_orig, node.parent, output, is_leaf=False, force_compute=force_compute)
        #     else:
        #         return self.evaluate(v_orig, node.reaction, output, node=node, is_leaf=is_leaf, force_compute=force_compute) + self.all_evaluate(0, node.parent, output, is_leaf=False, force_compute=force_compute)

    def __call__(self, v_orig : float, node : Node, output : str, force_compute=False):
        return self.all_evaluate(v_orig, node, output, force_compute=force_compute)