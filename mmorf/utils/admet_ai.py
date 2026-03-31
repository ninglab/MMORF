from .run_once import run_once
import os, json
from argparse import ArgumentParser
import torch, argparse, numpy
import numpy.core
torch.serialization.add_safe_globals([argparse.Namespace, numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtypes.Float64DType, numpy.dtype])

@run_once
def get_admet_model():
    from admet_ai import ADMETModel
    return ADMETModel()

CACHE_DIR = os.environ.get("mmorf_CACHE_DIR", os.path.dirname(__file__))
carc_cache = {}
if not os.environ.get("mmorf_NO_CACHE", False):
    if os.path.exists(CACHE_DIR + "/carcinogenicity_cache.json"):
        with open(CACHE_DIR + "/carcinogenicity_cache.json", "r") as f:
            data = f.read()
            carc_cache = json.loads(data) if "{" in data or "[" in data else {}

def save_carc_cache():
    if not os.environ.get("mmorf_NO_CACHE", False):
        with open(CACHE_DIR + "/carcinogenicity_cache.json", "w") as f:
            f.write(json.dumps(carc_cache))