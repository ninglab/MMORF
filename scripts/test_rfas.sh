#!/bin/bash
source scripts/test_all_setup.sh RFAS $1 $2

mkdir RFAS_ALL_$1

time python3 scripts/test_mas_sparse.py --device cuda:0 --auto 1 --output "RFAS_ALL_$1/output_${ID}.json" --context "$CONTEXT" "$SMILES"