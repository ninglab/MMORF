from .base import ReactionLevelWrapper, ValueFn
import hashlib
import os, json, itertools
import pandas as pd
from ..utils.tanimoto import tanimoto_similarity, save_fpsim_cache
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class Pyro(ReactionLevelWrapper):
    def child_type(self):
        return _Pyro

class _Pyro(ValueFn):
    """
    A value function for pyrophoricity.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []
        self.pyros = pd.read_csv(os.path.dirname(__file__) + "/pyrophoricity.csv")["SMILES"].tolist()

    # @profile
    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # optimistic because 0.0 indicates no pyrophoric
        # if output is None:
        #     output = "./"
        # unique_string = "_pyro_" + reaction
        # file = output + "_pyro_"+hashlib.sha256(unique_string.encode()).hexdigest()+".json"
        # if os.path.exists(file):
        #     with open(file, 'r') as f:
        #         data = json.load(f)
        #         score = data.get("score", 0.0)
        if True:
            smis = reaction.replace(">>", ".").split(".")
            if len(smis) < 1:
                score = 0.0
            else:
                sims = []
                for s1, s2 in itertools.product(smis, self.pyros):
                    try:
                        sim = tanimoto_similarity(s1, s2, sanitize=False)
                        sims.append(sim)
                    except Exception as e:
                        sims.append(0.0)
                score = max(sims)
                save_fpsim_cache()
                score = float(score)
                # with open(file, 'w') as f:
                #     json.dump({"score": score}, f)
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * Pyro() + {self.intercept}"