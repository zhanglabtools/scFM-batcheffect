# scFM: Single-cell Foundation Model Evaluation Framework

## Project Overview

**scFM** is a comprehensive single-cell foundation model evaluation pipeline designed to systematically assess the performance of different single-cell gene expression embedding models on 6 biomedical datasets.

### Core Features

- **Multi-Model Support**: Integrates 8 state-of-the-art single-cell foundation models
- **Multi-Dataset Evaluation**: Standardized evaluation on 6 multi-species, multi-tissue datasets
- **Complete Evaluation Pipeline**: End-to-end workflow from data preprocessing to result visualization
- **Batch Effect Correction**: Integrated evaluation with 3 batch correction methods
- **Upstream Task Evaluation**: Verify embedding quality through linear probing

---

## Project Structure

```
scFM/
├── Src/
│   ├── config_loader.py              # Unified configuration loader
│   ├── config.yaml                   # Central configuration file
│   ├── run_dataset.sh               # Dataset evaluation entry point
│   ├── run.sh                        # Project entry point
│   │
│   └── core_pipeline/               # 7-stage evaluation pipeline
│       ├── 0_data_preprocess/      # Stage 0: Data preprocessing & standardization
│       ├── 1_data_model_preparation/  # Stage 1: Model-specific data preparation
│       ├── 2_extract_embedding/    # Stage 2: Embedding extraction
│       ├── 3_integration/          # Stage 3: Embedding integration & visualization
│       ├── 4_benchmark/            # Stage 4: scIB metrics benchmarking
│       ├── 5_batch_correction/     # Stage 5: Batch effect correction
│       └── 6_probing/              # Stage 6: Linear probing analysis
│
├── Data/
│   ├── download/                    # Downloaded raw data
│   └── Evaluation/                  # Dataset preprocessing output
│
└── Result/                          # Final evaluation results
    └── {dataset_name}/
        ├── {model_name}/
        │   ├── Embeddings_{model_name}.h5ad
        │   ├── benchmark/           # scIB metrics results
        │   └── probing_original/    # Linear probing results
        └── metrics_summary.csv      # Aggregated results table

```

---

## Supported Models

### Embedding Models (7)

| Model Name | Dimension | Training Data Scale | Type |
|---------|------|-----------|------|
| **GeneFormer** | 1152 | 30M cells | Transformer |
| **GeneCompass** | 768 | 10M+ cells | BERT + multi-modal |
| **scFoundation** | 512 | Large-scale | Foundation Model |
| **UCE** | 1280 | Public datasets | Universal embedding |
| **CellPLM** | 768 | Pre-trained | PLM |
| **scCello** | 768 | Specific data | Autoencoder |
| **NicheFormer** | 768 | Spatial transcriptomics | Spatial-aware |

### Integration Methods (3)

- **PCA**: Principal Component Analysis
- **Harmony**: Batch-aware integration
- **Scanorama**: Panoramic integration

### Batch Correction Methods (3)

- **batch_cells**: Batch-level mean centering
- **batch_celltypes**: Cell type-level batch processing
- **pca_debias**: Principal component removal

---

## Supported Datasets

| Dataset | Samples | Cells | Species | Tissue/Type |
|------|------|------|------|---------|
| **limb** | 12 | ~100k | Human+Mouse | Limb development |
| **liver** | 8-10 | ~100k | Human | Liver |
| **immune** | Multiple | ~500k | Human | Immune cells |
| **HLCA** | 16 | ~660k | Human | Lung |
| **huaxi** | 5 | ~60k | Human | Clinical samples |
| **idtrack** | Multiple | Mixed | Mixed | Other |

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n scfm python=3.10
conda activate scfm

# Install dependencies
pip install scanpy anndata pandas numpy torch matplotlib scikit-learn pyyaml

# Model-specific dependencies (as needed)
pip install scib_metrics scvi-tools harmony scanorama
```

### 2. Configure config.yaml

Edit `Src/config.yaml`:

```yaml
model_paths:
  geneformer:
    model_dir: /path/to/geneformer/model
    code_path: /path/to/geneformer/code
    
datasets:
  limb:
    input_dir: /path/to/limb/data
    output_dir: /path/to/limb/output
```

### 3. Run the Evaluation Pipeline

#### Method 1: Run complete pipeline via run_dataset.sh

```bash
cd Src
./run_dataset.sh limb
```

#### Method 2: Run stage by stage

```bash
cd Src/core_pipeline

# Stage 0: Data preprocessing
python 0_data_preprocess/data_preprocess.py --dataset /path/to/limb

# Stage 1: Model preparation
for model in geneformer cellplm uce; do
    python 1_data_model_preparation/prepare_${model}.py --dataset /path/to/limb
done

# Stage 2: Embedding extraction
for model in geneformer cellplm uce; do
    python 2_extract_embedding/extract_embedding_${model}.py --dataset /path/to/limb
done

# Stage 3: Integration
cd 3_integration
./run_all.sh /path/to/limb

# Stage 4: Benchmarking
cd ../4_benchmark
./run_all.sh /path/to/limb

# Stage 5: Batch correction
cd ../5_batch_correction
./batch_normalize_run_all.sh /path/to/limb

# Stage 6: Linear probing
cd ../6_probing
python probing_main.py --dataset /path/to/limb --model geneformer
```

---

## Pipeline Stages Detailed Explanation

### Stage 0: Data Preprocessing (`0_data_preprocess`)

**Purpose**: Standardize heterogeneous h5ad files from CELLxGENE into a unified format

**Input**: Raw CELLxGENE h5ad files (downloaded via wget)

**Output**: Standardized `data.h5ad` containing:
- X: log1p normalized count matrix
- obs: Cell metadata (cell type, batch, donor, etc.)
- var: Gene metadata
- layers['counts']: Raw counts (for certain models)

**Dataset-specific logic**:

```python
def preprocess_limb():
    # Cross-species alignment (human+mouse)
    # Cell type annotation unification
    # Batch effect assessment

def preprocess_liver():
    # Quality control filtering
    # Cell type subset selection
    # Donor information preservation

def preprocess_immune():
    # Large-scale filtering
    # Cell type mapping to standard ontology
    # Donor stratification
```

**Usage**:
```bash
python data_preprocess.py --dataset /path/to/dataset
```

---

### Stage 1: Data Model Preparation (`1_data_model_preparation`)

**Purpose**: Prepare model-specific input formats for each model

**Standard processing**:

| Model | Processing | Output Format |
|------|------|--------|
| **GeneFormer** | Copy counts→X<br/>Add ensembl_id<br/>Add n_counts | h5ad |
| **GeneCompass** | Gene name→ID<br/>Z-score filtering<br/>Token encoding<br/>Log1p + Rank values | HuggingFace Dataset |
| **CellPLM** | Unique var_names | h5ad |
| **scFoundation** | Subset to 19264 genes<br/>Sparse matrix format | .npz |
| **UCE** | var_names→gene symbols | h5ad |
| **scCello** | Custom preprocessing | - |
| **NicheFormer** | Same as GeneFormer | h5ad |

**Usage**:
```bash
python prepare_geneformer.py --dataset /path/to/dataset
python prepare_genecompass.py --dataset /path/to/dataset
# ... etc
```

---

### Stage 2: Embedding Extraction (`2_extract_embedding`)

**Purpose**: Generate cell embeddings using pre-trained models

**Output location**: `Result/{dataset_name}/{model_name}/`

**File formats**:
- GeneFormer: `*_emb.csv` (batches)
- GeneCompass: `cell_embeddings.npy`
- scFoundation: `benchmark_*.npy`
- UCE: `adata_uce_*.h5ad`
- Others: Model-specific formats

**Usage**:
```bash
python extract_embedding_geneformer.py --dataset /path/to/dataset --gpu-id 0
python extract_embedding_genecompass.py --dataset /path/to/dataset
# ... etc
```

**GPU allocation**: Configure GPU ID for each model in config.yaml

---

### Stage 3: Integration (`3_integration`)

**Purpose**: Integrate model embeddings with original data and compute UMAP visualization

**Processing workflow**:
1. Load original `data.h5ad`
2. Load model-specific embeddings
3. Integrate into `adata.obsm['X_emb']`
4. Compute UMAP: `adata.obsm['X_umap']`
5. Save as `Embeddings_{model}.h5ad`

**Usage**:
```bash
cd 3_integration
./run_all.sh /path/to/dataset /path/to/result
```

**Processor**:
```python
python integrate.py --dataset /path/to/dataset --model geneformer --result-dir /path/to/result
```

---

### Stage 4: Benchmarking (`4_benchmark`)

**Purpose**: Evaluate embedding quality using scIB metrics

**Evaluation metrics**:

**Biological conservation**:
- NMI (Normalized Mutual Information): Cell type label preservation
- ARI (Adjusted Rand Index): Clustering preservation

**Batch effect removal**:
- Batch correction index: Batch mixing degree
- GraphConnectivity: Graph connectivity

**Technical metrics**:
- Silhouette score: Cell type separation degree
- Isolation index

**Usage**:
```bash
cd 4_benchmark
./run_all.sh /path/to/dataset /path/to/result
```

**Processor**:
```python
python benchmark.py --dataset /path/to/dataset --model geneformer --result-dir /path/to/result
```

---

### Stage 5: Batch Correction (`5_batch_correction`)

**Purpose**: Apply batch correction methods and re-evaluate

**Methods**:

1. **batch_cells** (Simple):
   ```
   X_corrected = X - batch_mean[batch_id]
   ```

2. **batch_celltypes** (Moderate):
   ```
   X_corrected = X - batch_celltype_mean[batch_id, celltype]
   ```

3. **pca_debias** (PCA debiasing):
   ```
   Remove top N principal components from X_emb
   ```

**Output**: `Embeddings_{model}_batch_corrected.h5ad` containing `X_emb_batch`

**Usage**:
```bash
cd 5_batch_correction
./batch_normalize_run_all.sh /path/to/dataset /path/to/result batch_cells
./batch_normalize_run_all.sh /path/to/dataset /path/to/result batch_celltypes
./batch_normalize_run_all.sh /path/to/dataset /path/to/result pca_debias
```

---

### Stage 6: Linear Probing (`6_probing`)

**Purpose**: Evaluate embeddings through 5-fold cross-validated linear classification

**Workflow**:
1. **Data splitting**: Create 5-fold CV splits
2. **Feature extraction**: Extract from `X_emb` or `X_emb_batch`
3. **Training**: Use logistic regression or SVM for each fold
4. **Evaluation**: Compute 4 classification metrics
5. **Aggregation**: Average results across 5 folds

**Classification tasks**:
- **Primary task**: Cell type classification
- **Batch task** (optional): Batch ID prediction
- **Supervised task** (optional): Other metadata

**Output**:
```
probing_original/
├── cv_results_summary.csv        # 5-fold average results
├── fold_{0-4}/
│   ├── predictions.csv           # Predictions
│   └── metrics.json              # Per-fold metrics
└── plots/
    ├── confusion_matrix.png
    └── roc_curves.png
```

**Usage**:
```bash
python probing_main.py --dataset /path/to/dataset --model geneformer
python probing_main.py --dataset /path/to/dataset --model geneformer --type batch_normalized
```

**CLI arguments**:
- `--dataset`: Dataset directory (required)
- `--model`: Model name, e.g., 'geneformer' (required)
- `--type`: 'original' or 'batch_normalized' (default: original)
- `--result-dir`: Result directory (default: auto-inferred)



## References

### Model Papers

- **GeneFormer**: https://www.nature.com/articles/s41467-023-43139-9
- **GeneCompass**: [Citation pending]
- **scFoundation**: https://www.biorxiv.org/content/10.1101/2023.11.28.568918
- **UCE**: [Citation]
- **scGPT**: https://www.nature.com/articles/s41467-024-45749-5

### scIB Benchmark

- Paper: https://www.nature.com/articles/s41592-021-01336-8
- Code: https://github.com/theislab/scib

### Data Sources

- CELLxGENE: https://cellxgene.cziscience.com/
- HLCA: https://www.tissue-atlas.org/

---

## License and Citation

If you use this project, please cite:

```bibtex
@software{scfm2025,
  title={scFM: Single-cell Foundation Model Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourrepo/scfm}
}
```

---

## Support and Contact

- **Bug Reports**: Submit GitHub Issues
- **Feature Requests**: Create GitHub Discussion
- **Technical Support**: [contact info]

---

## Changelog

### v1.0.0 (2025-11-19)

- ✅ Core pipeline implementation (7 stages)
- ✅ 8 model integration
- ✅ 6 dataset support
- ✅ scIB benchmarking
- ✅ Linear probing analysis
- ✅ Batch effect correction
- ✅ Central configuration system

### Known Issues

- [ ] NicheFormer GPU memory optimization
- [ ] scGPT multi-GPU support
- [ ] Large dataset streaming processing

---

**Last Updated**: 2025-11-19  
**Status**: Production Ready ✅
