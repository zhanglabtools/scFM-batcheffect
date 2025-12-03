#!/bin/bash
# Extract UCE embeddings using LCBERT
#
# UCE (Universal Cell Embeddings) is extracted using multi-GPU inference
# Input: {dataset_dir}/uce/adata.h5ad (from 0b_data_model_preparation)
# Output: {dataset_dir}/../Result/{dataset_name}/uce/embeddings.h5ad
#
# Requires: LCBERT UCE model + accelerate for multi-GPU (configured in config.yaml)
# GPU devices: Configure CUDA_VISIBLE_DEVICES as needed

set -e

DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

# Check input file
INPUT_FILE="$DATASET_DIR/uce/adata.h5ad"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: UCE h5ad not found at $INPUT_FILE"
    exit 1
fi

# Extract dataset name from path
DATASET_NAME=$(basename "$DATASET_DIR")

# Create output directory
RESULT_DIR="$DATASET_DIR/../Result/$DATASET_NAME/uce"
mkdir -p "$RESULT_DIR"

# Get UCE configuration from config.yaml
UCE_CFG=$(python3 << 'EOF'
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
            uce_cfg = config.get('model_paths', {}).get('uce', {})
            
            result = {
                'code_path': uce_cfg.get('code_path', "/home/wanglinting/LCBERT/Code/model-uce"),
                'model_path': uce_cfg.get('model_path', "/home/wanglinting/LCBERT/Code/model-uce/model_files/33l_8ep_1024t_1280.torch"),
                'gpu': uce_cfg.get('gpu', 3),
            }
            print(json.dumps(result))
            sys.exit(0)
        except:
            pass

# Fallback to defaults
result = {
    'code_path': "/home/wanglinting/LCBERT/Code/model-uce",
    'model_path': "/home/wanglinting/LCBERT/Code/model-uce/model_files/33l_8ep_1024t_1280.torch",
    'gpu': 3,
}
print(json.dumps(result))
EOF
)

# Parse config
UCE_CODE=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['code_path'])")
UCE_MODEL=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")
UCE_GPU=$(echo "$UCE_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu'])")

# Set GPU devices from config
export CUDA_VISIBLE_DEVICES=$UCE_GPU

# Change to UCE model directory
cd "$UCE_CODE"

echo "[UCE] Extracting embeddings from $INPUT_FILE"
echo "[UCE] Output directory: $RESULT_DIR"
echo "[UCE] Using GPU: $UCE_GPU"
echo "[UCE] Model path: $UCE_MODEL"

# Run UCE embedding extraction using accelerate for multi-GPU
# Parameters:
#   --adata_path: Path to input h5ad file
#   --dir: Output directory for results
#   --model_loc: Path to UCE model checkpoint
#   --species: Species (human or mouse)
#   --nlayers: Number of layers in model
#   --batch_size: Batch size for inference
#   --multi_gpu: Use multi-GPU processing

accelerate launch --multi_gpu --num_processes=1 eval_single_anndata.py \
    --adata_path "$INPUT_FILE" \
    --dir "$RESULT_DIR" \
    --model_loc "$UCE_MODEL" \
    --species human \
    --nlayers 33 \
    --batch_size 32 \
    --multi_gpu false

echo "[UCE] ✓ Embedding extraction complete"
echo "[UCE] Results saved to $RESULT_DIR"
