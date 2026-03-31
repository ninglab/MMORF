import torch
import numpy as np
import json
from json import JSONEncoder

# Patch to undo the "weights_only" default in torch.load introduced in PyTorch 2.1, which breaks loading some saved models.
torch._load = torch.load
def load(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch._load(*args, **kwargs)
torch.load = load

# Patch for VisibleDeprecationWarning in numpy (versions differ with ADMET-AI and other dependencies, trying to unify)
class VisibleDeprecationWarning(Warning):
    pass
np.VisibleDeprecationWarning = VisibleDeprecationWarning

# Set up the JSON default encoder to handle custom objects
JSONEncoder._old_default = JSONEncoder.default
def json_default_encoder(self, obj):
    try:
        return JSONEncoder._old_default(self, obj)
    except:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return str(obj)
JSONEncoder.default = json_default_encoder