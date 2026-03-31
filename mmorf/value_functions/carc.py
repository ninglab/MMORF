from .base import ValueFn, ReactionLevelWrapper
import hashlib
import os, json, re
from ..utils.bbsa import bbsa_alerts
profile = lambda x: x  # Default to no profiling if not enabled
if os.environ.get("ENABLE_PROFILING", "0") == "1":
    from line_profiler import profile as p
    profile = p

class FastCarc(ReactionLevelWrapper):
    def child_type(self):
        return _FastCarc
    
class SlowCarc(ReactionLevelWrapper):
    def child_type(self):
        return _SlowCarc

class _SlowCarc(ValueFn):
    """
    A value function for synthesis planning that evaluates the carcinogenicity
    of molecules using the ADMET AI model or a rule-based approach.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        from ..utils.admet_ai import get_admet_model, carc_cache, save_carc_cache
        self.model = get_admet_model()
        self.children = []

    def __call__(self, reaction=None, v_orig=None, output=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # optimistic because 0.0 indicates no carcinogenicity
        # if output is None:
        #     output = "./"
        # unique_string = "_carc_" + reaction
        # file = output + "_carc_"+hashlib.sha256(unique_string.encode()).hexdigest()+".json"
        score = 0.0
        # if os.path.exists(file):
        #     with open(file, 'r') as f:
        #         data = json.load(f)
        #         score = data.get("score", 0.0)
        # else:
        if True:
            smis = set(reaction.replace(">>", ".").split("."))
            if len(smis) > 0:
                from ..utils.admet_ai import get_admet_model, carc_cache, save_carc_cache
                smis = smis - set(carc_cache.keys())
                all_carc = self.model.predict(list(smis) + ["C", "O"])["Carcinogens_Lagunin"].to_dict()# placeholder for actual carcinogenicity score
                carc_cache.update(all_carc)
                save_carc_cache()
                score = max([carc_cache[s] for s in smis] + [0])
                score = float(score)
            # with open(file, 'w') as f:
            #     json.dump({"score": score}, f)
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * Carc() + {self.intercept}"
    
class _FastCarc(ValueFn):
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # not a reaction, so no carcinogenicity evaluation.
        smis = set(reaction.replace(">>", ".").split("."))
        alerts = set().union(*[bbsa_alerts(s) for s in smis])
        if len(alerts) > 0:
            return self.coeff * 1.0 + self.intercept
        else:
            return self.intercept