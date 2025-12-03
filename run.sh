#!/bin/bash
# run.sh - Main pipeline orchestrator
# Runs complete evaluation for specified datasets
# Usage: bash run.sh [DATASET | all]
#   bash run.sh limb              # Run single dataset
#   bash run.sh all               # Run all datasets
#   bash run.sh                   # Show usage

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   scFM Embedding Evaluation Pipeline                      ║"
echo "║   Complete workflow: aggregate → model prep → extract →   ║"
echo "║   integrate → benchmark → normalize → probing             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# If no argument provided, show usage
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name> | all"
    echo ""
    bash "$SCRIPT_DIR/run_dataset.sh"
    exit 1
fi

# Run dataset pipeline
bash "$SCRIPT_DIR/run_dataset.sh" "$1"
