#!/bin/bash

set -e

# Set datasets and models   
DATA_LIST="limb,liver,Immune,HLCA_assay,HLCA_disease,HLCA_sn"
MODEL_LIST="uce,cellplm,geneformer,genecompass,scfoundation,sccello,nicheformer"
CONFIG_FILE="config.yaml"

IFS=',' read -ra DATASETS <<< "$DATA_LIST"
IFS=',' read -ra MODELS <<< "$MODEL_LIST"

# ============================================================================
# STAGE 0: Data Preprocessing
# ============================================================================
echo "Stage 0: Data Preprocessing"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    echo "  $dataset..."
    python3 core_pipeline/0_data_preprocess/data_preprocess.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
done

# ============================================================================
# STAGE 1: Model Preparation (all models)
# ============================================================================
echo ""
echo "Stage 1: Model Preparation"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    echo "  $dataset..."
    python3 core_pipeline/1_data_model_preparation/prepare_geneformer.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
    python3 core_pipeline/1_data_model_preparation/prepare_uce.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
    python3 core_pipeline/1_data_model_preparation/prepare_cellplm.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
    python3 core_pipeline/1_data_model_preparation/prepare_scfoundation.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
    python3 core_pipeline/1_data_model_preparation/prepare_genecompass.py \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
    bash core_pipeline/1_data_model_preparation/prepare_sccello.sh \
        --dataset "$dataset" --config "$CONFIG_FILE" || exit 1
done

# ============================================================================
# STAGE 2: Embedding Extraction
# ============================================================================
echo ""
echo "Stage 2: Embedding Extraction"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    for model in "${MODELS[@]}"; do
        model=${model// /}
        echo "  $dataset → $model..."
        
        case $model in
            uce)
                bash core_pipeline/2_extract_embedding/extract_embedding_UCE.sh \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            cellplm)
                python3 core_pipeline/2_extract_embedding/extract_embedding_cellplm.py \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            geneformer)
                python3 core_pipeline/2_extract_embedding/extract_embedding_geneformer.py \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            genecompass)
                python3 core_pipeline/2_extract_embedding/extract_embedding_genecompass.py \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            scfoundation)
                bash core_pipeline/2_extract_embedding/extract_embedding_scFoundation.sh \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            sccello)
                bash core_pipeline/2_extract_embedding/extract_embedding_scCello.sh \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
            nicheformer)
                python3 core_pipeline/2_extract_embedding/extract_embedding_nicheformer.py \
                    --dataset "$dataset" --config "$CONFIG_FILE" || exit 1 ;;
        esac
    done
done

# ============================================================================
# STAGE 3: Integration
# ============================================================================
echo ""
echo "Stage 3: Integration"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    for model in "${MODELS[@]}"; do
        model=${model// /}
        echo "  $dataset → integrate $model..."
        python3 core_pipeline/3_integration/integrate.py \
            --dataset "$dataset" --model "$model" --config "$CONFIG_FILE" || exit 1
    done
done

# ============================================================================
# STAGE 4: Benchmarking
# ============================================================================
echo ""
echo "Stage 4: Benchmarking"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    for model in "${MODELS[@]}"; do
        model=${model// /}
        echo "  $dataset → benchmark $model..."
        python3 core_pipeline/4_benchmark/benchmark.py \
            --dataset "$dataset" --model "$model" --config "$CONFIG_FILE" || exit 1
    done
done

# ============================================================================
# STAGE 5: Batch Correction
# ============================================================================
echo ""
echo "Stage 5: Batch Correction"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    for model in "${MODELS[@]}"; do
        model=${model// /}
        echo "  $dataset → batch correct $model..."
        python3 core_pipeline/5_batch_correction/batch_normalize.py \
            --dataset "$dataset" --model "$model" --config "$CONFIG_FILE" || exit 1
    done
done


# ============================================================================
# STAGE 6: Probing Analysis
# ============================================================================
echo ""
echo "Stage 6: Probing Analysis"
for dataset in "${DATASETS[@]}"; do
    dataset=${dataset// /}
    for model in "${MODELS[@]}"; do
        model=${model// /}
        echo "  $dataset → probing $model (original)..."
        python3 core_pipeline/6_probing/probing.py \
            --dataset "$dataset" --model "$model" --config "$CONFIG_FILE" || exit 1
        
        echo "  $dataset → probing $model (batch-corrected)..."
        python3 core_pipeline/6_probing/probing.py \
            --dataset "$dataset" --model "$model" --batch-center --config "$CONFIG_FILE" || exit 1
    done
done

echo ""
echo "✓ All stages complete"