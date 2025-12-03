#!/bin/bash
# run_dataset.sh - Unified dataset evaluation pipeline
# 
# Reads dataset configuration from config.yaml and runs complete evaluation pipeline
# 
# Usage:
#   bash run_dataset.sh limb              # Run limb dataset evaluation
#   bash run_dataset.sh liver             # Run liver dataset evaluation
#   bash run_dataset.sh HLCA_disease      # Run HLCA disease evaluation
#   bash run_dataset.sh [all]             # Run all datasets sequentially

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
CORE_PIPELINE_DIR="$SCRIPT_DIR/core_pipeline"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config.yaml not found at $CONFIG_FILE"
    exit 1
fi

# Function to parse YAML and extract value
get_yaml_value() {
    local file=$1
    local dataset=$2
    local key=$3
    
    python3 << EOF
import yaml
with open('$file', 'r') as f:
    config = yaml.safe_load(f)
    
try:
    value = config['datasets']['$dataset']['$key']
    if isinstance(value, list):
        print(' '.join(str(v) for v in value))
    else:
        print(value)
except (KeyError, TypeError):
    print("")
EOF
}

# Function to run pipeline for a single dataset
run_dataset_pipeline() {
    local dataset_name=$1
    
    echo "=========================================="
    echo "Running evaluation pipeline for: $dataset_name"
    echo "=========================================="
    
    # Get configuration values from config.yaml
    output_dir=$(get_yaml_value "$CONFIG_FILE" "$dataset_name" "output_dir")
    batch_key=$(get_yaml_value "$CONFIG_FILE" "$dataset_name" "batch_key")
    celltype_key=$(get_yaml_value "$CONFIG_FILE" "$dataset_name" "celltype_key")
    models=$(get_yaml_value "$CONFIG_FILE" "$dataset_name" "models")
    
    if [ -z "$output_dir" ]; then
        echo "ERROR: Dataset '$dataset_name' not found in datasets.yaml"
        return 1
    fi
    
    echo "Output directory: $output_dir"
    echo "Batch key: $batch_key"
    echo "Celltype key: $celltype_key"
    echo "Models: $models"
    
    # Ensure output directory exists
    mkdir -p "$output_dir"
    
    # Create config file in output directory (where other scripts expect it)
    CONFIG_FILE="$output_dir/config.yaml"
    cat > "$CONFIG_FILE" << EOFCONFIG
# Auto-generated config for dataset: $dataset_name
dataset_name: "$dataset_name"
output_dir: "$output_dir"
batch_key: "$batch_key"
celltype_key: "$celltype_key"
pre_integrated_embedding_obsm_key: "X_pca"

models:
$(echo "$models" | tr ' ' '\n' | sed 's/^/  - /')
EOFCONFIG
    
    echo "Config file created: $CONFIG_FILE"
    
    # Step 0: Data preprocessing
    echo ""
    echo "[STEP 0] Data preprocessing..."
    python "$CORE_PIPELINE_DIR/0_data_preprocess/data_preprocess.py" \
        --dataset "$dataset_name" \
        --config "$CONFIG_FILE" \
        --output "$output_dir/data.h5ad" \
        || { echo "ERROR: Data preprocessing failed"; return 1; }
    
    echo "✓ Data preprocessing completed"
    
    # Step 1: Model-specific input preparation
    echo ""
    echo "[STEP 1] Model input preparation..."
    bash "$CORE_PIPELINE_DIR/1_data_model_preparation/run_all.sh" "$output_dir" \
        || { echo "WARNING: Some model inputs failed to prepare"; }
    
    echo "✓ Model input preparation completed"
    
    # Step 2: Extract embeddings
    echo ""
    echo "[STEP 2] Extracting embeddings from all models..."
    bash "$CORE_PIPELINE_DIR/2_extract_embedding/run_all.sh" "$output_dir" \
        || { echo "WARNING: Some embeddings failed to extract"; }
    
    echo "✓ Embedding extraction completed"
    
    # Step 3: Integration
    echo ""
    echo "[STEP 3] Integrating embeddings..."
    bash "$CORE_PIPELINE_DIR/3_integration/run_all.sh" "$output_dir" \
        || { echo "WARNING: Integration step had issues"; }
    
    echo "✓ Integration completed"
    
    # Step 4: Benchmark
    echo ""
    echo "[STEP 4] Running benchmark evaluation..."
    bash "$CORE_PIPELINE_DIR/4_benchmark/run_all.sh" "$output_dir" \
        || { echo "WARNING: Benchmark step had issues"; }
    
    echo "✓ Benchmark evaluation completed"
    
    # Step 5: Batch correction
    echo ""
    echo "[STEP 5] Batch correction..."
    bash "$CORE_PIPELINE_DIR/5_batch_correction/batch_normalize_run_all.sh" "$output_dir" \
        || { echo "WARNING: Batch correction had issues"; }
    
    echo "✓ Batch correction completed"
    
    # Step 6: Probing
    echo ""
    echo "[STEP 6] Probing analysis..."
    bash "$CORE_PIPELINE_DIR/6_probing/run_all.sh" "$output_dir" \
        || { echo "WARNING: Probing analysis had issues"; }
    
    echo "✓ Probing analysis completed"
    
    echo ""
    echo "=========================================="
    echo "✓ Pipeline completed for: $dataset_name"
    echo "Results saved to: $output_dir"
    echo "=========================================="
    
    return 0
}

# Main logic
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name> | all"
    echo ""
    echo "Available datasets:"
    python3 << EOF
import yaml
with open('$DATASETS_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
    for dataset in config['datasets']:
        desc = config['datasets'][dataset].get('description', '')
        print(f"  • {dataset:20} - {desc}")
EOF
    exit 1
fi

if [ "$1" = "all" ]; then
    echo "Running all datasets..."
    datasets=("limb" "liver" "Immune" "HLCA_assay" "HLCA_disease" "HLCA_sn")
    
    failed=()
    for dataset in "${datasets[@]}"; do
        if run_dataset_pipeline "$dataset"; then
            echo ""
        else
            failed+=("$dataset")
            echo ""
        fi
    done
    
    if [ ${#failed[@]} -gt 0 ]; then
        echo "❌ Failed datasets: ${failed[*]}"
        exit 1
    else
        echo "✅ All datasets completed successfully!"
    fi
else
    # Run single dataset
    run_dataset_pipeline "$1"
fi
