from .base import ReactionLevelWrapper, ValueFn
import hashlib
import os, json, re
import numpy as np
# from ..utils.coprinet_apptainer import cost_estimate
from ..utils.purch_list import purch_cost
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class BBPrice(ReactionLevelWrapper):
    def child_type(self):
        return _BBPrice

class _BBPrice(ValueFn):
    """
    A value function for predicted cost of purchasable reactants.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    @profile
    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # optimistic because 0.0 indicates no cost
        reactants = reaction.split(">>")[0].strip().split(".")
        value = 0.0
        for p in reactants:
            value += purch_cost.get(p, np.exp(4.37))
        return self.coeff * value + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * BBPrice() + {self.intercept}"