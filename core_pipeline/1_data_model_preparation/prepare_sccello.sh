#!/bin/bash
# Prepare scCello model input via LCBERT wrapper
#
# scCello is a wrapper around LCBERT's data transformation script
# Input: {dataset_dir}/uce/adata.h5ad (UCE format with counts)
# Output: {dataset_dir}/sccello/ (processed by LCBERT script)
#
# External Dependency: LCBERT installation configured in config.yaml

set -e

DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

echo "[scCello] Preparing input from $DATASET_DIR"

# Input file (UCE format)
INPUT_FILE="$DATASET_DIR/uce/adata.h5ad"
if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] UCE h5ad not found at $INPUT_FILE"
    exit 1
fi

# Get LCBERT script path from config.yaml using Python
LCBERT_SCRIPT=$(python3 << 'EOF'
import os
import sys
import yaml

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
            script = config.get('model_paths', {}).get('sccello', {}).get('transformation_script')
            if script:
                print(script)
                sys.exit(0)
        except:
            pass

# Fallback to hardcoded default
print("/home/wanglinting/LCBERT/Code/model-sccello/sccello/script/run_data_transformation.py")
EOF
)

if [ ! -f "$LCBERT_SCRIPT" ]; then
    echo "[ERROR] LCBERT scCello script not found at $LCBERT_SCRIPT"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$DATASET_DIR/sccello"
mkdir -p "$OUTPUT_DIR"

# Run LCBERT transformation script
# Note: Script expects input h5ad and outputs to specified directory
echo "[scCello] Running LCBERT transformation..."
python "$LCBERT_SCRIPT" \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --species human

echo "[scCello] Transformation complete. Output saved to $OUTPUT_DIR"
