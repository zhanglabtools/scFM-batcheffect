#!/bin/bash
# Prepare model-specific inputs from data.h5ad
#
# Dependencies:
# - data.h5ad (from 0a_data_aggregation)
# - Generates UCE format first (dependency for GeneCompass, scCello)
# - Then generates other model formats in parallel

set -e

# Source directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_DIR="${1:-.}"

echo "=========================================="
echo "Preparing model-specific inputs"
echo "Dataset directory: $DATASET_DIR"
echo "=========================================="

# Check if data.h5ad exists
if [ ! -f "$DATASET_DIR/data.h5ad" ]; then
    echo "ERROR: data.h5ad not found in $DATASET_DIR"
    exit 1
fi

# ============================================================
# Step 1: Prepare UCE format (dependency for GeneCompass/scCello)
# ============================================================
echo ""
echo "Step 1: Preparing UCE format..."
python "$SCRIPT_DIR/prepare_uce.py" --dataset "$DATASET_DIR"

# ============================================================
# Step 2: Prepare other formats (CellPLM, GeneFormer, scFoundation)
# These don't depend on UCE
# ============================================================
echo ""
echo "Step 2: Preparing CellPLM, GeneFormer, scFoundation..."

python "$SCRIPT_DIR/prepare_cellplm.py" --dataset "$DATASET_DIR"
echo "✓ CellPLM ready"

python "$SCRIPT_DIR/prepare_geneformer.py" --dataset "$DATASET_DIR"
echo "✓ GeneFormer ready"

python "$SCRIPT_DIR/prepare_scfoundation.py" --dataset "$DATASET_DIR"
echo "✓ scFoundation ready"

# ============================================================
# Step 3: Prepare GeneCompass (depends on UCE)
# ============================================================
echo ""
echo "Step 3: Preparing GeneCompass (depends on UCE)..."
python "$SCRIPT_DIR/prepare_genecompass.py" --dataset "$DATASET_DIR" --species human
echo "✓ GeneCompass ready"

# ============================================================
# Step 4: Prepare scCello (depends on UCE)
# ============================================================
echo ""
echo "Step 4: Preparing scCello (depends on UCE)..."
bash "$SCRIPT_DIR/prepare_sccello.sh" "$DATASET_DIR"
echo "✓ scCello ready"

echo ""
echo "=========================================="
echo "✓ All model formats prepared successfully"
echo "=========================================="

