from .base import ValueFn
import hashlib
import os, json, re
import numpy as np
# from ..utils.coprinet_apptainer import cost_estimate
from ..utils.purch_list import purch_cost

class Depth(ValueFn):
    """
    A value function for the depth of a reaction tree.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        return 1.0 + (node.depth if node is not None else 0.0)
    
    def __repr__(self):
        return f"{self.coeff} * Depth() + {self.intercept}"