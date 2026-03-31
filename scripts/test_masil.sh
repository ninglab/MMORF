#!/bin/bash

source scripts/test_all_setup.sh MASIL $1 $2
echo "==== MASIL ===="
echo "Using LLM: $REPLANNER_LLM"

mkdir MASIL_ALL_$1

time python3 scripts/test_masil.py --device cuda:0 --auto 20 --output "MASIL_ALL_$1/output_new5_${ID}.json" --context "$CONTEXT" "$SMILES"