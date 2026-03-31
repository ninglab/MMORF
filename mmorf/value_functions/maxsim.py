from .base import ValueFn, ReactionLevelWrapper
import hashlib
import os, json, re
from ..utils.tanimoto import tanimoto_similarity, save_fpsim_cache
import itertools
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class MaxSim(ReactionLevelWrapper):
    def child_type(self):
        return _MaxSim

class _MaxSim(ValueFn):
    """
    A value function for synthesis planning that always returns 1.
    This is a placeholder and should be replaced with a more meaningful implementation.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.smiles = args[0:]
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    @profile
    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # default 0.0 indicates no similarity
        mols = reaction.replace(">>", ".").split(".")
        sims = [0.0] # default to no similarity
        for s1, s2 in itertools.product(mols, self.smiles):
            try:
                sim = tanimoto_similarity(s1, s2, sanitize=True)
                sims.append(sim)
            except Exception as e:
                print(e)
                pass
        score = max(sims)
        score = float(score)
        save_fpsim_cache()
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * MaxSim() + {self.intercept}"