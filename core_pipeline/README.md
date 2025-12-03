# Single-Cell Embedding Evaluation Pipeline

Complete end-to-end pipeline for evaluating single-cell embedding models and integration methods across multiple datasets.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Pipeline (6 Stages)                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  0_data_preprocess          → Standardize input data          │
│         ↓                                                     │
│  1_data_model_preparation   → Prepare model-specific inputs   │
│         ↓                                                     │
│  2_extract_embedding        → Extract embeddings (12 models)  │
│         ↓                                                     │
│  3_integration              → Integrate embeddings            │
│         ↓                                                     │
│  4_benchmark                → Evaluate integration quality    │
│         ↓                                                     │
│  5_batch_correction         → Apply batch correction (3 ways) │
│         ↓                                                     │
│  6_probing                  → Downstream classification tasks │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Structure

### Stage 0: Data Preprocessing
**Directory**: `0_data_preprocess/`

Standardizes input data into unified HDF5 format.

**Main Script**: `aggregate_data.py`

**Inputs**: 
- Raw data files (various formats: h5ad, h5, csv, etc.)

**Outputs**: 
- `data.h5ad` - Standardized dataset with metadata

**Usage**:
```bash
python aggregate_data.py --input <raw_data> --output data.h5ad
```

---

### Stage 1: Model Data Preparation
**Directory**: `1_data_model_preparation/`

Prepares dataset in specific formats required by different models.

**Scripts**:
- `prepare_uce.py` - UCE (Universal Cell Embedding)
- `prepare_cellplm.py` - CellPLM (Cell Language Model)
- `prepare_geneformer.py` - GeneFormer (Gene Transformer)
- `prepare_genecompass.py` - GeneCompass
- `prepare_scfoundation.py` - scFoundation
- `prepare_sccello.sh` - scCello
- `run_all.sh` - Orchestrate all preparations

**Inputs**: 
- `data.h5ad` from Stage 0

**Outputs**: 
- Model-specific format files (directories/NPY/CSV/etc.)

**Usage**:
```bash
# Prepare all models
bash run_all.sh /path/to/dataset /path/to/data.h5ad

# Prepare single model
python prepare_geneformer.py --dataset /path/to/dataset --data /path/to/data.h5ad
```

---

### Stage 2: Embedding Extraction
**Directory**: `2_extract_embedding/`

Extracts embeddings using 8 different models via inference/pretrained weights.

**Models Supported**:
- **UCE**: Universal Cell Embedding (1280-dim)
- **CellPLM**: Cell Language Model (768-dim)
- **GeneFormer**: Gene Transformer (1152-dim)
- **GeneCompass**: Gene Compass (768-dim)
- **scFoundation**: Foundation model (512-dim)
- **scCello**: scCello (768-dim)
- **NicheFormer**: Spatial niche modeling (768-dim)
- **scGPT**: Large language model (768-dim)

**Main Script**: `run_all.sh`

**Inputs**: 
- Prepared data from Stage 1
- `config.yaml` with model list

**Outputs**: 
- `Embeddings_*.h5ad` files (one per model)

**Usage**:
```bash
# Extract all embeddings listed in config.yaml
bash run_all.sh /path/to/dataset /path/to/Result/dataset
```

---

### Stage 3: Integration
**Directory**: `3_integration/`

Combines embeddings from multiple models using 4 integration methods.

**Integration Methods**:
- **PCA**: Principal Component Analysis (baseline)
- **scVI**: Variational Autoencoder integration
- **Harmony**: Batch correction algorithm
- **Scanorama**: Panoramic integration

**Main Script**: `integrate.py`

**Inputs**: 
- Individual embeddings from Stage 2
- `config.yaml` with model list

**Outputs**: 
- `Embeddings_{model}_integrated.h5ad` (with X_umap computed)
- UMAP visualizations

**Usage**:
```bash
python integrate.py --dataset /path/to/dataset --model geneformer
bash run_all.sh /path/to/dataset /path/to/Result/dataset
```

---

### Stage 4: Benchmark Evaluation
**Directory**: `4_benchmark/`

Evaluates integration quality using comprehensive scIB metrics.

**Metrics Evaluated**:
- **Biological Conservation** (60% weight):
  - NMI (Normalized Mutual Information)
  - ARI (Adjusted Rand Index)  
  - Silhouette Score
  
- **Batch Correction** (40% weight):
  - BRAS (Batch Removal After Scaling)
  - kBET (k-nearest Batch Effect Test)
  - Graph Connectivity
  - PCR (Percent Variance Retained)

**Main Script**: `benchmark.py`

**Inputs**: 
- Integrated embeddings from Stage 3
- `config.yaml` metadata keys

**Outputs**: 
- `metrics_{model}.csv` - Benchmark results per model

**Usage**:
```bash
python benchmark.py --dataset /path/to/dataset --model geneformer
bash run_all.sh /path/to/dataset /path/to/Result/dataset
```

---

### Stage 5: Batch Correction
**Directory**: `5_batch_correction/`

Applies batch effect correction using 3 different methods.

**Correction Methods**:
1. **batch_cells**: Subtract batch-wise mean embedding
2. **batch_celltypes**: Subtract batch-wise celltype-mean
3. **pca_debias**: Remove top principal components

**Main Script**: `batch_normalize.py`

**Inputs**: 
- Integrated embeddings from Stage 3
- Benchmark results from Stage 4 (for reference)

**Outputs**: 
- `Embeddings_{model}_batch_corrected.h5ad` 
- `metrics_batch_corrected.csv` - Benchmark on corrected embedding
- UMAP visualizations

**Usage**:
```bash
# Single model, single method
python batch_normalize.py --dataset /path/to/dataset --model geneformer --method batch_cells

# All models with specific method
bash batch_normalize_run_all.sh /path/to/dataset /path/to/Result/dataset batch_cells
```

---

### Stage 6: Probing (Downstream Classification)
**Directory**: `6_probing/`

Evaluates embeddings on downstream classification tasks using multiple classification methods.

**Scripts**:
- `probing_analysis.py` - Evaluate original embeddings
- `probing_analysis_batch_normalize.py` - Evaluate batch-corrected embeddings
- `probing_data_split.py` - Prepare train/test splits
- `utils_cv.py` - Cross-validation utilities
- `run_all.sh` - Orchestration

**Classifiers Used**:
- Logistic Regression
- SVM
- Random Forest
- Neural Network (MLP)
- KNN

**Metrics**:
- Accuracy
- F1-score
- Balanced Accuracy
- Macro/Micro Precision/Recall

**Main Script**: `run_all.sh`

**Inputs**: 
- Integrated embeddings from Stage 3
- Batch-corrected embeddings from Stage 5
- `config.yaml` metadata

**Outputs**: 
- `probing_results.csv` - Classification results
- Per-classifier performance metrics

**Usage**:
```bash
# Run probing on original embeddings
python probing_analysis.py --dataset /path/to/dataset --model geneformer

# Run probing on batch-corrected embeddings
python probing_analysis_batch_normalize.py --dataset /path/to/dataset --model geneformer
```

---

## Configuration

All stages use `config.yaml` in the dataset directory to define:

```yaml
batch_key: organism              # Batch effect column (e.g., organism, batch)
celltype_key: cell_type          # Cell type label column
pre_integrated_embedding_obsm_key: X_pca  # Reference embedding for benchmarking

models:                          # List of models to evaluate
  - geneformer
  - genecompass
  - scfoundation
  - sccello
  - nicheformer
  - scgpt
  - uce
  - cellplm
  - pca
  - scvi
  - harmony
  - scanorama
```

See `datasets/` folder for example configs per dataset.

---

## Quick Start

### Example Workflow: Evaluate on Limb Dataset

```bash
cd Src/core_pipeline

# Stage 0: Aggregate data
python 0_data_preprocess/aggregate_data.py \
    --input ../../LCBERT/Data/newdata/limb \
    --output datasets/limb/data.h5ad

# Stage 1: Prepare model-specific data
bash 1_data_model_preparation/run_all.sh datasets/limb datasets/limb/data.h5ad

# Stage 2: Extract embeddings
bash 2_extract_embedding/run_all.sh datasets/limb ../../Result/limb

# Stage 3: Integrate embeddings
bash 3_integration/run_all.sh datasets/limb ../../Result/limb

# Stage 4: Benchmark integration
bash 4_benchmark/run_all.sh datasets/limb ../../Result/limb

# Stage 5: Batch correction
bash 5_batch_correction/batch_normalize_run_all.sh datasets/limb ../../Result/limb batch_cells

# Stage 6: Probing tasks
bash 6_probing/run_all.sh datasets/limb ../../Result/limb
```

Or use the main `run.sh` script for complete pipeline.

---

## File Organization

```
core_pipeline/
├── 0_data_preprocess/           # Stage 0: Data aggregation
│   ├── aggregate_data.py
│   └── [supporting files]
│
├── 1_data_model_preparation/    # Stage 1: Model preparation
│   ├── prepare_*.py
│   ├── prepare_*.sh
│   └── run_all.sh
│
├── 2_extract_embedding/         # Stage 2: Embedding extraction
│   ├── extract_embedding_*.py
│   ├── extract_embedding_*.sh
│   ├── run_all.sh
│   └── [supporting files]
│
├── 3_integration/               # Stage 3: Integration
│   ├── integrate.py
│   └── run_all.sh
│
├── 4_benchmark/                 # Stage 4: Benchmarking
│   ├── benchmark.py
│   └── run_all.sh
│
├── 5_batch_correction/          # Stage 5: Batch correction
│   ├── batch_normalize.py
│   └── batch_normalize_run_all.sh
│
├── 6_probing/                   # Stage 6: Probing tasks
│   ├── probing_analysis.py
│   ├── probing_analysis_batch_normalize.py
│   ├── probing_data_split.py
│   ├── utils_cv.py
│   └── run_all.sh
│
├── datasets/                    # Dataset configurations
│   ├── config_template.yaml
│   ├── limb/
│   ├── liver/
│   ├── Immune/
│   ├── HLCA_assay/
│   ├── HLCA_disease/
│   └── HLCA_sn/
│
└── README.md                    # This file
```

---

## Supported Models & Methods

### Embedding Models (Stage 2)
- **8 foundation models**: UCE, CellPLM, GeneFormer, GeneCompass, scFoundation, scCello, NicheFormer, scGPT
- **Output dimensions**: 512-1280D depending on model

### Integration Methods (Stage 3)
- **4 algorithms**: PCA, scVI, Harmony, Scanorama
- **All combinations**: 8 models × 4 methods = 32 integrated embeddings per dataset

### Batch Correction Methods (Stage 5)
- **3 approaches**: Batch cells, Batch celltypes, PCA debiasing
- **Evaluation**: Full scIB benchmark on corrected embeddings

### Downstream Tasks (Stage 6)
- **5 classifiers**: Logistic Regression, SVM, Random Forest, MLP, KNN
- **Metrics**: Accuracy, F1, Balanced Accuracy, Precision, Recall

---

## Performance Notes

| Stage | Time/Model | Memory | Parallelizable |
|-------|-----------|--------|-----------------|
| 0 Data Preprocess | ~5-10 min | 2-4 GB | Per dataset |
| 1 Model Prep | ~2-5 min | 2-4 GB | Per model |
| 2 Extraction | ~10-30 min | 4-8 GB | Per model |
| 3 Integration | ~2-5 min | 2-4 GB | Per method |
| 4 Benchmark | ~5-15 min | 8-16 GB | 4 cores |
| 5 Batch Correct | ~1-5 min | 4-8 GB | Per method |
| 6 Probing | ~5-10 min | 2-4 GB | Per task |

**Total end-to-end**: ~2-4 hours per dataset with full model suite

---

## Dependencies

- Python 3.8+
- scanpy, anndata, numpy, pandas, matplotlib
- scikit-learn, scipy
- scib-metrics, scib
- Model-specific libraries (transformers, scgpt, cellplm, etc.)

See `requirements.txt` for complete list.

---

## Key Outputs Per Dataset

After running all 6 stages, you get:

1. **Individual embeddings**: `Embeddings_*.h5ad` (12 files)
2. **Integration benchmarks**: `metrics_*.csv` (per model & method)
3. **Batch-corrected embeddings**: `Embeddings_*_batch_corrected.h5ad`
4. **Batch correction benchmarks**: `metrics_batch_corrected.csv`
5. **Probing results**: Classification accuracy across tasks
6. **Visualizations**: UMAP plots for QC at each stage

---

## Troubleshooting

**Issue**: "config.yaml not found"
- **Solution**: Create `datasets/{dataset_name}/config.yaml` with required keys

**Issue**: "Model weights not found"
- **Solution**: Run Stage 1 to download/prepare model data first

**Issue**: Memory errors during benchmarking
- **Solution**: Reduce `n_jobs` parameter or run on subset of models

**Issue**: GPU out of memory
- **Solution**: Set `CUDA_VISIBLE_DEVICES` or use CPU (slower)

---

## Citation & References

This pipeline integrates:
- scIB metrics ([Garcia-Alonso et al., 2021](https://www.nature.com/articles/s41592-021-01136-0))
- Multiple state-of-art embedding models and integration methods
- Comprehensive benchmarking framework

---

## License

Same as parent project

## Contact

For questions or issues, refer to individual stage documentation or contact the development team.
