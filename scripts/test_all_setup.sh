#!/bin/bash

source .env

export REPLANNER_CACHE_DIR=.
export REPLANNER_DEBUG=0

export REPLANNER_LLM=${2:-"Qwen30B"}
export ID="ALL_${3}_${REPLANNER_LLM}"
export REPLANNER_LOG_FILE="run_multi_new${1}__${ID}.jsonl"
export mmorf_CACHE_DIR="cache/new${1}__${ID}"
mkdir -p $mmorf_CACHE_DIR

export FAST_CARCINOGENICITY=0

export SMILES=$(sed -n "${3}p" data/${DATASET_PREFIX}_products.txt) # remove _round2 for full run
export TASK=$(sed -n "${3}p" data/${DATASET_PREFIX}_tasks.txt) # remove _round2 for full run


if [ "$3" -lt 112 ]; then
    export CONTEXT="$TASK Only worry about the explicitly specified constraints in the prompt.  Consider only molecules listed as IARC Group 1, 2A, or 2B to be carcinogens, if applicable to the prompt. Additionally, do not worry about molecules that are known to be safe (IARC Group 3 or unlisted), even if they are flagged as possibly carcinogenic by the models, unless you sincerely believe them to be potential carcinogens based on your chemistry expertise."
fi

if [ "$3" = "Qwen30B" ]; then
    export CONTEXT="$CONTEXT Do not concern yourself with feasibility or plausibility of the chemical reactions.  You must assume all reactions are feasible and all routes are plausible as written."
fi
