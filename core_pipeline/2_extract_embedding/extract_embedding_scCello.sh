#!/bin/bash
# Extract scCello embeddings
#
# scCello embeddings are extracted using HuggingFace-compatible zero-shot model
# Input: {dataset_dir}/sccello/ (processed data from 0b_data_model_preparation)
# Output: {dataset_dir}/../Result/{dataset_name}/sccello/embeddings/cell_embeddings.npy
#
# Requires: scCello pretrained model from HuggingFace (configured in config.yaml)

set -e

DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

# Check input directory
INPUT_DIR="$DATASET_DIR/sccello"
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: scCello data directory not found at $INPUT_DIR"
    exit 1
fi

# Extract dataset name from path
DATASET_NAME=$(basename "$DATASET_DIR")

# Create output directory
RESULT_DIR="$DATASET_DIR/../Result/$DATASET_NAME/sccello"
mkdir -p "$RESULT_DIR"

# Get scCello configuration from config.yaml
SCCELLO_CFG=$(python3 << 'EOF'
import os
import sys
import yaml
import json

# Try to load config.yaml
config_paths = [
    os.path.join(os.path.dirname(__file__), '../../config.yaml'),
    '/home/wanglinting/scFM/Src/config.yaml',
]

for config_path in config_paths:
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            sccello_cfg = config.get('model_paths', {}).get('sccello', {})
            
            result = {
                'code_path': sccello_cfg.get('code_path', "/home/wanglinting/LCBERT/Code/model-sccello"),
                'pretrained_ckpt': sccello_cfg.get('pretrained_ckpt', "/home/wanglinting/LCBERT/Download/scCello/scCello-zeroshot"),
                'gpu': sccello_cfg.get('gpu', 5),
            }
            print(json.dumps(result))
            sys.exit(0)
        except:
            pass

# Fallback to defaults
result = {
    'code_path': "/home/wanglinting/LCBERT/Code/model-sccello",
    'pretrained_ckpt': "/home/wanglinting/LCBERT/Download/scCello/scCello-zeroshot",
    'gpu': 5,
}
print(json.dumps(result))
EOF
)

# Parse config
SCCELLO_CODE=$(echo "$SCCELLO_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['code_path'])")
SCCELLO_PRETRAINED=$(echo "$SCCELLO_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['pretrained_ckpt'])")
SCCELLO_GPU=$(echo "$SCCELLO_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu'])")

# Set GPU device
export CUDA_VISIBLE_DEVICES=$SCCELLO_GPU

# Change to scCello model directory
cd "$SCCELLO_CODE"

echo "[scCello] Extracting embeddings from $INPUT_DIR"
echo "[scCello] Output directory: $RESULT_DIR"
echo "[scCello] Using GPU: $SCCELLO_GPU"

# Run scCello embedding extraction
# Parameters:
#   --pretrained_ckpt: Path to pretrained model or HuggingFace model name
#   --data_path: Path to processed data (output from run_data_transformation.py)
#   --output_dir: Output directory for embeddings
#   --batch_size: Inference batch size
#   --data_format: Input data format (huggingface, pickle, numpy)
#   --filename_prefix: Output file name prefix

python extract_embeddings.py \
    --pretrained_ckpt "$SCCELLO_PRETRAINED" \
    --data_path "$INPUT_DIR/proceseed_pretraining_data_postproc" \
    --output_dir "$RESULT_DIR" \
    --batch_size 32 \
    --data_format huggingface

echo "[scCello] ✓ Embedding extraction complete"
echo "[scCello] Results saved to $RESULT_DIR"
