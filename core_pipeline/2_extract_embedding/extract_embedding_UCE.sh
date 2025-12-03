#!/bin/bash
# Extract UCE embeddings using LCBERT
#
# UCE (Universal Cell Embeddings) is extracted using multi-GPU inference
# Input: {output_data_dir}/uce/adata.h5ad
# Output: {output_res_dir}/uce/embeddings.h5ad
#
# Requires: LCBERT UCE model + accelerate for multi-GPU (configured in config.yaml)

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

# Get UCE configuration from config.yaml
UCE_CFG=$(python3 << 'EOF'
import os
import sys
import yaml
import json

config_path = sys.argv[1]
dataset_name = sys.argv[2]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = config['datasets'][dataset_name]
uce_cfg = config.get('model_paths', {}).get('uce', {})

data_dir = dataset_config['output_data_dir']
res_dir = dataset_config.get('output_res_dir', os.path.join(data_dir, 'Result', dataset_name))

result = {
    'data_dir': data_dir,
    'res_dir': res_dir,
    'code_path': uce_cfg.get('code_path', ""),
    'model_path': uce_cfg.get('model_path', ""),
    'gpu': uce_cfg.get('gpu', 3),
    'nlayers': uce_cfg.get('nlayers', 33),
    'batch_size': uce_cfg.get('batch_size', 32),
    'species': uce_cfg.get('species', 'human'),
}
print(json.dumps(result))
EOF
"$CONFIG" "$DATASET"
)

# Parse config
DATA_DIR=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['data_dir'])")
RES_DIR=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['res_dir'])")
UCE_CODE=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['code_path'])")
UCE_MODEL=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")
UCE_GPU=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu'])")
UCE_NLAYERS=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['nlayers'])")
UCE_BATCH_SIZE=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['batch_size'])")
UCE_SPECIES=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['species'])")

INPUT_FILE="$DATA_DIR/uce/adata.h5ad"
RESULT_DIR="$RES_DIR/uce"

# Check input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: UCE h5ad not found at $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$RESULT_DIR"

# Set GPU devices from config
export CUDA_VISIBLE_DEVICES=$UCE_GPU

# Change to UCE model directory
cd "$UCE_CODE"

echo "[UCE] Dataset: $DATASET"
echo "[UCE] Input: $INPUT_FILE"
echo "[UCE] Output directory: $RESULT_DIR"
echo "[UCE] Using GPU: $UCE_GPU"
echo "[UCE] Model path: $UCE_MODEL"
echo "[UCE] Species: $UCE_SPECIES"
echo "[UCE] Layers: $UCE_NLAYERS"
echo "[UCE] Batch size: $UCE_BATCH_SIZE"

# Run UCE embedding extraction using accelerate for multi-GPU
accelerate launch --multi_gpu --num_processes=1 eval_single_anndata.py \
    --adata_path "$INPUT_FILE" \
    --dir "$RESULT_DIR" \
    --model_loc "$UCE_MODEL" \
    --species "$UCE_SPECIES" \
    --nlayers "$UCE_NLAYERS" \
    --batch_size "$UCE_BATCH_SIZE" \
    --multi_gpu false

echo "[UCE] ✓ Embedding extraction complete"
echo "[UCE] Results saved to $RESULT_DIR"