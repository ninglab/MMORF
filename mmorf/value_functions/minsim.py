from .base import ValueFn
import hashlib
import os, json
from ..utils.tanimoto import tanimoto_similarity, save_fpsim_cache
import itertools
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class MinSim(ValueFn):
    """
    A wrapper for the _MinSim value function to be used at the reaction level.
    """
    def child_type(self):
        return _MinSim

class _MinSim(ValueFn):
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
            return self.intercept + self.coeff * 1.0 # default 0.0 high similarity
        if output is None:
            output = "./"
        mols = reaction.replace(">>", ".").split(".")
        unique_string = "_minsim_" + reaction
        file = output + "_minsim_"+hashlib.sha256(unique_string.encode()).hexdigest()+".json"
        if os.path.exists(file):
            with open(file, 'r') as f:
                data = json.load(f)
                score = data.get("score", 1.0)
        else:
            sims = [] # default to no similarity
            for s1, s2 in itertools.product(mols, self.smiles):
                try:
                    sim = tanimoto_similarity(s1, s2, sanitize=True)
                    sims.append(sim)
                except Exception as e:
                    pass
            score = min(sims) if len(sims) > 0 else 0.0
            score = float(score)
            save_fpsim_cache()
            with open(file, 'w') as f:
                json.dump({"score": score}, f)
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * MinSim() + {self.intercept}"