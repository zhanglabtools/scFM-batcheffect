#!/bin/bash
# Extract scFoundation embeddings
#
# scFoundation embeddings are extracted using the get_embedding.py script
# Input: {dataset_dir}/scfoundation/data.npz (from 0b_data_model_preparation)
# Output: {dataset_dir}/../Result/{dataset_name}/scfoundation/cell_embeddings.npy
#
# Requires: scFoundation model checkpoint (configured in config.yaml)

set -e

DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

# Check input file
INPUT_FILE="$DATASET_DIR/scfoundation/data.npz"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: scFoundation data.npz not found at $INPUT_FILE"
    exit 1
fi

# Extract dataset name from path
DATASET_NAME=$(basename "$DATASET_DIR")

# Create output directory
RESULT_DIR="$DATASET_DIR/../Result/$DATASET_NAME/scfoundation"
mkdir -p "$RESULT_DIR"

# Get scFoundation configuration from config.yaml
SCFOUNDATION_CFG=$(python3 << 'EOF'
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
            scfoundation_cfg = config.get('model_paths', {}).get('scfoundation', {})
            
            result = {
                'code_path': scfoundation_cfg.get('code_path', "/home/wanglinting/LCBERT/Code/model-scFoundation"),
                'model_path': scfoundation_cfg.get('model_path', "/home/wanglinting/LCBERT/Download/scFoundation-main/model/models/models.ckpt"),
                'gpu': scfoundation_cfg.get('gpu', 4),
            }
            print(json.dumps(result))
            sys.exit(0)
        except:
            pass

# Fallback to defaults
result = {
    'code_path': "/home/wanglinting/LCBERT/Code/model-scFoundation",
    'model_path': "/home/wanglinting/LCBERT/Download/scFoundation-main/model/models/models.ckpt",
    'gpu': 4,
}
print(json.dumps(result))
EOF
)

# Parse config
SCFOUNDATION_CODE=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['code_path'])")
SCFOUNDATION_MODEL=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['model_path'])")
SCFOUNDATION_GPU=$(echo "$SCFOUNDATION_CFG" | python3 -c "import sys, json; print(json.load(sys.stdin)['gpu'])")

# Set GPU device from config
export CUDA_VISIBLE_DEVICES=$SCFOUNDATION_GPU

# Change to scFoundation model directory
cd "$SCFOUNDATION_CODE"

echo "[scFoundation] Extracting embeddings from $INPUT_FILE"
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
