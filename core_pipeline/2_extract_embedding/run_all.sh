#!/bin/bash
# Extract embeddings from all models
#
# Orchestrates embedding extraction for all 6 supported models:
# - UCE (universal cell embeddings)
# - CellPLM (foundation model)
# - GeneFormer (transformer-based, requires tokenized input)
# - scFoundation (sparse matrix format)
# - GeneCompass (multi-modal prior knowledge)
# - NicheFormer (spatial-aware model)
#
# Execution order: Can run in parallel (no dependencies)
# Each model uses different GPU device to avoid memory conflicts

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_DIR="$1"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset_dir>"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

echo "=========================================="
echo "Extracting embeddings from all models"
echo "Dataset directory: $DATASET_DIR"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track success/failure
FAILED_MODELS=()
SUCCESS_MODELS=()

# ============================================================
# UCE Embedding Extraction
# ============================================================
echo "[UCE] Starting embedding extraction..."
if bash "$SCRIPT_DIR/extract_embedding_UCE.sh" "$DATASET_DIR"; then
    echo -e "${GREEN}✓ UCE embeddings extracted${NC}"
    SUCCESS_MODELS+=("UCE")
else
    echo -e "${RED}✗ UCE extraction failed${NC}"
    FAILED_MODELS+=("UCE")
fi
echo ""

# ============================================================
# GeneFormer Embedding Extraction
# ============================================================
echo "[GeneFormer] Starting embedding extraction..."
if python "$SCRIPT_DIR/extract_embedding_geneformer.py" --dataset "$DATASET_DIR" --gpu 0 --batch-size 32; then
    echo -e "${GREEN}✓ GeneFormer embeddings extracted${NC}"
    SUCCESS_MODELS+=("GeneFormer")
else
    echo -e "${RED}✗ GeneFormer extraction failed${NC}"
    FAILED_MODELS+=("GeneFormer")
fi
echo ""

# ============================================================
# scFoundation Embedding Extraction
# ============================================================
echo "[scFoundation] Starting embedding extraction..."
if bash "$SCRIPT_DIR/extract_embedding_scFoundation.sh" "$DATASET_DIR"; then
    echo -e "${GREEN}✓ scFoundation embeddings extracted${NC}"
    SUCCESS_MODELS+=("scFoundation")
else
    echo -e "${RED}✗ scFoundation extraction failed${NC}"
    FAILED_MODELS+=("scFoundation")
fi
echo ""

# ============================================================
# scCello Embedding Extraction
# ============================================================
echo "[scCello] Starting embedding extraction..."
if bash "$SCRIPT_DIR/extract_embedding_scCello.sh" "$DATASET_DIR"; then
    echo -e "${GREEN}✓ scCello embeddings extracted${NC}"
    SUCCESS_MODELS+=("scCello")
else
    echo -e "${RED}✗ scCello extraction failed${NC}"
    FAILED_MODELS+=("scCello")
fi
echo ""

# ============================================================
# GeneCompass Embedding Extraction
# ============================================================
echo "[GeneCompass] Starting embedding extraction..."
if python "$SCRIPT_DIR/extract_embedding_genecompass.py" --dataset "$DATASET_DIR" --gpu 1 --batch-size 64; then
    echo -e "${GREEN}✓ GeneCompass embeddings extracted${NC}"
    SUCCESS_MODELS+=("GeneCompass")
else
    echo -e "${RED}✗ GeneCompass extraction failed${NC}"
    FAILED_MODELS+=("GeneCompass")
fi
echo ""

# ============================================================
# NicheFormer Embedding Extraction
# ============================================================
echo "[NicheFormer] Starting embedding extraction..."
if python "$SCRIPT_DIR/extract_embedding_nicheformer.py" --dataset "$DATASET_DIR" --gpu 2 --batch-size 32; then
    echo -e "${GREEN}✓ NicheFormer embeddings extracted${NC}"
    SUCCESS_MODELS+=("NicheFormer")
else
    echo -e "${RED}✗ NicheFormer extraction failed${NC}"
    FAILED_MODELS+=("NicheFormer")
fi
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "Embedding extraction summary"
echo "=========================================="
echo ""

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo -e "${GREEN}Successfully extracted:${NC}"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "  ✓ $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo -e "${RED}Failed to extract:${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
    echo ""
    echo -e "${YELLOW}Check logs for details. Some models may require specific GPU resources.${NC}"
fi

echo ""
echo "Output directory structure:"
echo "  {dataset}/../Result/{dataset_name}/"
echo "  ├── uce/                 (UCE results)"
echo "  ├── geneformer/          (GeneFormer results)"
echo "  ├── scfoundation/        (scFoundation results)"
echo "  ├── sccello/             (scCello results)"
echo "  ├── genecompass/         (GeneCompass results)"
echo "  └── nicheformer/         (NicheFormer results)"
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All embeddings extracted successfully!${NC}"
    exit 0
fi
