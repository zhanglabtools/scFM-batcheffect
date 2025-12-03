#!/bin/bash
# Prepare scCello model input 
#
# Input: {output_data_dir}/uce/adata.h5ad (UCE format with counts)
# Output: {output_data_dir}/sccello/ (processed by LCBERT script)
#
# Usage: ./prepare_sccello.sh --dataset limb --config /path/to/config.yaml

set -e

DATASET=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        *) echo "Usage: $0 --dataset <name> --config <path>"; exit 1 ;;
    esac
done

[ -z "$DATASET" ] || [ -z "$CONFIG_FILE" ] && echo "Missing arguments" && exit 1
[ ! -f "$CONFIG_FILE" ] && echo "Config file not found" && exit 1

# Parse YAML helper
parse_yaml() {
    python3 -c "
import yaml
with open('$1') as f:
    config = yaml.safe_load(f)
keys = '$2'.split('.')
for key in keys:
    config = config[key]
print(config)
"
}

OUTPUT_DIR=$(parse_yaml "$CONFIG_FILE" "datasets.$DATASET.output_data_dir")
TRANS_SCRIPT=$(parse_yaml "$CONFIG_FILE" "model_paths.sccello.transformation_script")
INPUT_FILE="$OUTPUT_DIR/uce/adata.h5ad"
SCCELLO_OUTPUT_DIR="$OUTPUT_DIR/sccello"

[ ! -f "$INPUT_FILE" ] && echo "UCE input not found: $INPUT_FILE" && exit 1
[ ! -f "$TRANS_SCRIPT" ] && echo "Transformation script not found: $TRANS_SCRIPT" && exit 1

mkdir -p "$SCCELLO_OUTPUT_DIR"

echo "Processing $DATASET..."
python3 "$TRANS_SCRIPT" \
    --input_file "$INPUT_FILE" \
    --output_dir "$SCCELLO_OUTPUT_DIR" \
    --species human

echo "Done: $SCCELLO_OUTPUT_DIR"