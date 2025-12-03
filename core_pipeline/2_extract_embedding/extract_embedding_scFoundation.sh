#!/bin/bash
# Extract scFoundation embeddings
#
# scFoundation embeddings are extracted using the get_embedding.py script
# Input: {output_data_dir}/scfoundation/data.npz
# Output: {output_res_dir}/scfoundation/cell_embeddings.npy
#
# Requires: scFoundation model checkpoint (configured in config.yaml)

set -e

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

# Get scFoundation configuration from config.yaml
SCFOUNDATION_CFG=$(python3 << 'EOF'
import os
import sys
import yaml
import json

config_path = sys.argv[1]
dataset_name = sys.argv[2]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = config['datasets'][dataset_name]
scfoundation_cfg = config.get('model_paths', {}).get('scfoundation', {})

data_dir = dataset_config['output_data_dir']
res_dir = dataset_config.get('output_res_dir', os.path.join(data_dir, 'Result', dataset_name))

result = {
    'data_dir': data_dir,
    'res_dir': res_dir,
    'code_path': scfoundation_cfg.get('code_path', ""),
    'model_path': scfoundation_cfg.get('model_path', ""),
    'gpu': scfoundation_cfg.get('gpu', 4),
}
print(json.dumps(result))
EOF
"$CONFIG" "$DATASET"
)

# Parse config
DATA_DIR=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['data_dir'])")
RES_DIR=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['res_dir'])")
SCFOUNDATION_CODE=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['code_path'])")
SCFOUNDATION_MODEL=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")
SCFOUNDATION_GPU=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu'])")

INPUT_FILE="$DATA_DIR/scfoundation/data.npz"
RESULT_DIR="$RES_DIR/scfoundation"

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: scFoundation data.npz not found at $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$RESULT_DIR"

# Set GPU device from config
export CUDA_VISIBLE_DEVICES=$SCFOUNDATION_GPU

# Change to scFoundation model directory
cd "$SCFOUNDATION_CODE"

echo "[scFoundation] Dataset: $DATASET"
echo "[scFoundation] Input: $INPUT_FILE"
echo "[scFoundation] Output directory: $RESULT_DIR"
echo "[scFoundation] Using GPU: $SCFOUNDATION_GPU"
echo "[scFoundation] Model path: $SCFOUNDATION_MODEL"

# Run scFoundation embedding extraction
# Parameters:
#   --task_name: Task type (benchmark)
#   --input_type: Input data type (singlecell)
#   --output_type: Output type (cell)
#   --pool_type: Pooling type (all)
#   --tgthighres: Target high resolution (f1)
#   --pre_normalized: Data is pre-normalized (F=False)
#   --version: Model version (ce=core expression)
#   --data_path: Path to input NPZ file
#   --save_path: Output directory
#   --model_path: Path to model checkpoint

python get_embedding.py \
    --task_name benchmark \
    --input_type singlecell \
    --output_type cell \
    --pool_type all \
    --tgthighres f1 \
    --pre_normalized F \
    --version ce \
    --data_path "$INPUT_FILE" \
    --save_path "$RESULT_DIR" \
    --model_path "$SCFOUNDATION_MODEL"

echo "[scFoundation] ✓ Embedding extraction complete"
echo "[scFoundation] Results saved to $RESULT_DIR"