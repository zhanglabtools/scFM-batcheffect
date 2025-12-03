#!/bin/bash
# Extract scCello embeddings
#
# scCello embeddings are extracted using HuggingFace-compatible zero-shot model
# Input: {output_data_dir}/sccello/ (processed data from prepare_sccello.sh)
# Output: {output_res_dir}/sccello/cell_embeddings.npy
#
# Requires: scCello pretrained model from HuggingFace (configured in config.yaml)

set -e

# Parse arguments
DATASET=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Usage: $0 --dataset <name> --config <path>"; exit 1 ;;
    esac
done

[ -z "$DATASET" ] || [ -z "$CONFIG" ] && echo "Error: Missing arguments" && exit 1
[ ! -f "$CONFIG" ] && echo "Error: Config file not found: $CONFIG" && exit 1

# Get paths and config from YAML
read -r DATA_DIR RES_DIR CODE_PATH PRETRAINED_CKPT GPU BATCH_SIZE TRANSFORMATION_SCRIPT <<< $(python3 << 'PYSCRIPT'
import sys
import os
import yaml
import json

config_path = sys.argv[1]
dataset_name = sys.argv[2]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = config['datasets'][dataset_name]
sccello_config = config['model_paths']['sccello']

data_dir = dataset_config['output_data_dir']
res_dir = dataset_config.get('output_res_dir')
code_path = sccello_config['code_path']
pretrained_ckpt = sccello_config['pretrained_ckpt']
gpu = sccello_config.get('gpu', 5)
batch_size = sccello_config.get('batch_size', 32)
transformation_script = sccello_config.get('transformation_script', '')

print(f"{data_dir} {res_dir} {code_path} {pretrained_ckpt} {gpu} {batch_size} {transformation_script}")
PYSCRIPT
python3 - "$CONFIG" "$DATASET"
)

INPUT_DIR="$DATA_DIR/sccello"
RESULT_DIR="$RES_DIR/sccello"

# Check input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: scCello data directory not found at $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$RESULT_DIR"

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU

# Change to scCello model directory
cd "$CODE_PATH"

echo "Dataset: $DATASET"
echo "Input dir: $INPUT_DIR"
echo "Output dir: $RESULT_DIR"
echo "GPU: $GPU"
echo "Batch size: $BATCH_SIZE"

# Check if input data exists
if [ ! -d "$INPUT_DIR/processed_pretraining_data_postproc" ]; then
    echo "ERROR: Processed data not found at $INPUT_DIR/processed_pretraining_data_postproc"
    echo "Make sure prepare_sccello.sh has been executed first"
    exit 1
fi

echo "Extracting scCello embeddings..."

# Run scCello embedding extraction
python extract_embeddings.py \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --data_path "$INPUT_DIR/processed_pretraining_data_postproc" \
    --output_dir "$RESULT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --data_format huggingface

echo "✓ scCello embedding extraction complete"
echo "Results saved to $RESULT_DIR"