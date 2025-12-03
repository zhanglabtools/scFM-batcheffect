#!/bin/bash
# Batch normalization orchestration script
# Reads models from config.yaml and applies batch correction

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_DIR="$1"
RESULT_DIR="${2:-}"
METHOD="${3:-batch_cells}"  # batch_cells, batch_celltypes, pca_debias

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir> [result_dir] [method]"
    echo ""
    echo "Methods:"
    echo "  batch_cells       - Subtract batch-wise mean (default)"
    echo "  batch_celltypes   - Subtract batch-wise celltype-mean"
    echo "  pca_debias        - Remove top principal components"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Check for config.yaml
CONFIG_FILE="$DATASET_DIR/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config.yaml not found at $CONFIG_FILE"
    exit 1
fi

# Infer result directory if not provided
if [ -z "$RESULT_DIR" ]; then
    DATASET_NAME=$(basename "$DATASET_DIR")
    RESULT_DIR="$DATASET_DIR/../Result/$DATASET_NAME"
fi

if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: Result directory not found: $RESULT_DIR"
    exit 1
fi

echo "=========================================="
echo "Batch Normalization from config.yaml"
echo "=========================================="
echo "Dataset: $DATASET_DIR"
echo "Config: $CONFIG_FILE"
echo "Result: $RESULT_DIR"
echo "Method: $METHOD"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Read models from config.yaml using Python
MODELS=$(python3 << 'EOF'
import yaml
import sys

config_file = sys.argv[1]
try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if isinstance(config, dict) and 'models' in config:
        models = config['models']
        if isinstance(models, list):
            print(' '.join(models))
        elif isinstance(models, str):
            print(models)
    else:
        print("")
except Exception as e:
    print(f"ERROR: Failed to read config: {e}", file=sys.stderr)
    sys.exit(1)
EOF
"$CONFIG_FILE")

if [ -z "$MODELS" ]; then
    echo "ERROR: No 'models' key found in config.yaml"
    exit 1
fi

# Convert to array
IFS=' ' read -ra MODEL_ARRAY <<< "$MODELS"

echo "Models to normalize: ${MODEL_ARRAY[@]}"
echo "Total: ${#MODEL_ARRAY[@]} models"
echo ""

FAILED_MODELS=()
SUCCESS_MODELS=()

# Track start time
start_time=$(date +%s)

# Process each model
for i in "${!MODEL_ARRAY[@]}"; do
    model="${MODEL_ARRAY[$i]}"
    model_num=$((i + 1))
    total=${#MODEL_ARRAY[@]}
    
    echo -e "${BLUE}[${model_num}/${total}]${NC} Processing: ${YELLOW}${model}${NC}"
    
    if python "$SCRIPT_DIR/batch_normalize.py" \
        --dataset "$DATASET_DIR" \
        --model "$model" \
        --method "$METHOD" \
        --result-dir "$RESULT_DIR"; then
        
        echo -e "${GREEN}✓${NC} ${model} completed"
        SUCCESS_MODELS+=("$model")
    else
        echo -e "${RED}✗${NC} ${model} failed"
        FAILED_MODELS+=("$model")
    fi
    echo ""
done

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_min=$((runtime / 60))
runtime_sec=$((runtime % 60))

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo -e "${GREEN}✓ Successfully normalized (${#SUCCESS_MODELS[@]}):${NC}"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "  • $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}✗ Failed (${#FAILED_MODELS[@]}):${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  • $model"
    done
fi

echo ""
echo "Runtime: ${runtime_min}m ${runtime_sec}s"
echo "=========================================="

# Exit with error if any failed
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All models batch-normalized successfully!${NC}"
    exit 0
fi
