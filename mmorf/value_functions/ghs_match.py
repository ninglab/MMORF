from .base import ValueFn, ReactionLevelWrapper
import hashlib
import os, json, re
from ..utils.pubchem import get_ghs_from_pubchem

class GHS(ReactionLevelWrapper):
    def child_type(self):
        return _GHS

class _GHS(ValueFn):
    """
    A value function for synthesis planning that evaluates the GHS hazard statements of reactants and products.
    This function checks for GHS hazard codes and returns a score based on the presence of hazardous substances.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.hazards = set(args[0:])
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    def __call__(self, reaction=None, v_orig=None, output=None, node=None, force_compute=False, **kwargs):
        if reaction is None:
            return self.intercept + self.coeff * 0.0 # optimistic because 0.0 indicates no GHS hazards
        if output is None:
            output = "./"
        # unique_string = "_ghs_" + reaction
        # file = output + "_ghs_"+hashlib.sha256(unique_string.encode()).hexdigest()+".json"
        # if os.path.exists(file):
        #     with open(file, 'r') as f:
        #         data = json.load(f)
        #         score = data.get("score", 0.0)
        # else:
        if True:
            reactants, products = reaction.split(">>")
            reactants = reactants.split(".")
            products = products.split(".")
            hazards = set()
            for reactant in reactants:
                hazards.update(get_ghs_from_pubchem(reactant))
            for product in products:
                hazards.update(get_ghs_from_pubchem(product))
            # Calculate score based on the presence of hazardous substances
            score = 0.0
            for hazard in self.hazards:
                if hazard in hazards:
                    score = 1.0
            score = float(score)
            # with open(file, 'w') as f:
            #     json.dump({"score": score}, f)
        return self.coeff * score + self.intercept
    
    def __repr__(self):
        return f"""{self.coeff} * GHS({', '.join(['"'+h+'"' for h in self.hazards])}) + {self.intercept}"""