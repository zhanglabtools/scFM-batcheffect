# Batch Effects Remain a Fundamental Barrier to Universal Embeddings in Single-Cell Foundation Models


## Abstract
Constructing a cell universe requires integrating heterogeneous single-cell RNA-seq datasets, which is hindered by diverse batch effects. Single-cell foundation models (scFMs), inspired by large language models, aim to learn universal cellular embeddings from large-scale single-cell data. However, unlike language, single-cell data are sparse, noisy, and strongly affected by batch artifacts, limiting cross-dataset transferability. Our systematic evaluation across diverse batch scenarios reveals that current scFMs fail to intrinsically remove batch effects, with batch signals persisting in pretrained embeddings. Post-hoc batch-centering partially improves alignment, highlighting the need for future scFMs to integrate explicit batch-effect correction mechanisms to achieve true universal cellular embeddings.

---

## About This Repository

This repository stores relevant analysis and plotting codes.

```
scFM-batcheffect/
├── config.yaml                    # Configuration file
├── run.sh                         # Master pipeline (all stages, all datasets)
├── README.md                      # This file
| 
├── figures/                       # Figure plotting Jupyter files 
|
└── core_pipeline/                 # Evaluation pipeline
    ├── 0_data_preprocess/         # Step 0: Data preprocessing & standardization
    ├── 1_data_model_preparation/  # Step 1: Model-specific data preparation
    ├── 2_extract_embedding/       # Step 2: Embedding extraction
    ├── 3_integration/             # Step 3: Embedding integration & visualization
    ├── 4_benchmark/               # Step 4: scIB metrics benchmarking
    ├── 5_batch_correction/        # Step 5: Batch-centering correction
    └── 6_probing/                 # Step 6: Linear probing analysis
```

---

## Resources of Models and Datasets

### Foundation Models
- [scGPT](https://github.com/bowang-lab/scGPT) 
- [GeneFormer](https://huggingface.co/ctheodoris/Geneformer) 
- [GeneCompass](https://github.com/xCompass-AI/GeneCompass)
- [scFoundation](https://github.com/biomap-research/scFoundation)
- [UCE](https://github.com/snap-stanford/UCE)
- [CellPLM](https://github.com/OmicsML/CellPLM)
- [scCello](https://github.com/DeepGraphLearning/scCello)
- [NicheFormer](https://github.com/theislab/nicheformer)

### Integration Methods
All the traditional integration methods can be applied using [scIB](https://github.com/theislab/scib).
 - PCA
 - [Harmony](https://github.com/immunogenomics/harmony)
 - [Scanorama](https://github.com/brianhie/scanorama)  
 - [scVI](https://scvi-tools.org/)  


### Datasets
 - [limb](https://cellxgene.cziscience.com/collections/4fefa187-5d14-4f1e-915b-c892ed320aab) 
 - [liver](https://cellxgene.cziscience.com/collections/ff69f0ee-fef6-4895-9f48-6c64a68c8289) 
 - [Immune](https://cellxgene.cziscience.com/collections/cc431242-35ea-41e1-a100-41e0dec2665b) 
 - [HLCA](https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293) 


---

## Usage

### 1. Environment Setup

#### Base Environment

First, create a conda environment with common dependencies:

```bash
# Create conda environment
conda create -n scfm python=3.10
conda activate scfm

# Install common dependencies
pip install scanpy pandas scipy numpy matplotlib scikit-learn pyyaml 

# Install evaluation and integration methods
pip install scib scib_metrics scvi-tools harmony-pytorch scanorama
```

#### Foundation Model-Specific Environment Setup

Each foundation model has different dependencies and require separate environments. Please refer to the respective tutorials of each model.


### 2. Configure config.yaml

Edit `config.yaml` to set up paths for your datasets and models:

```yaml

# Model directory paths
model_paths:
  geneformer:
    code_path: "/path/to/geneformer/code"
    model_path: "/path/to/geneformer/model"
    gpu: 0
    batch_size: 32

  scfoundation:
    code_path: "/path/to/scfoundation/code"
    model_path: "/path/to/scfoundation/model"
    gpu: 1
    batch_size: 32

  # ... other models

# Datasets
datasets:
  limb:
    data_path: "/path/to/limb/raw.h5ad"
    output_data_dir: "/path/to/limb/processed_data"
    output_res_dir: "/path/to/limb/results"
    batch_key: "batch"
    celltype_key: "cell_type"

  liver:
    data_path: "/path/to/liver/raw.h5ad"
    output_data_dir: "/path/to/liver/processed_data"
    output_res_dir: "/path/to/liver/results"
    batch_key: "batch"
    celltype_key: "cell_type"

  # ... other datasets

# Probing configuration
probing:
  n_splits: 5
  max_workers: 4

# Batch correction configuration
batch_correction:
  batch_cells:
    max_cells_per_batch: 10000
    random_seed: 42
    normalize: true
```

### 3. Run the Evaluation Pipeline

#### Option A: Run Full Pipeline (All Stages, All Datasets)

```bash
bash run.sh
```

This will execute all 6 stages sequentially for all datasets and models.

#### Option B: Run Pipeline Step by Step (Recommended)

Because deploying multiple foundation models in the same environment is challenging, it's best to run stages one by one, switching between model-specific environments as needed.

##### Stage 0: Data Preprocessing

Run once for all datasets:

```bash
cd core_pipeline

# Preprocess limb dataset
python 0_data_preprocess/data_preprocess.py \
    --dataset limb \
    --config ../config.yaml

# Preprocess liver datasets
python 0_data_preprocess/data_preprocess.py \
    --dataset liver \
    --config ../config.yaml

# Repeat for other datasets
```

##### Stage 1: Model-Specific Data Preparation

Run each model for each dataset:

```bash
# For limb dataset, prepare all model-specific data
python 1_data_model_preparation/prepare_geneformer.py --dataset limb --config ../config.yaml
python 1_data_model_preparation/prepare_uce.py --dataset limb --config ../config.yaml
python 1_data_model_preparation/prepare_cellplm.py --dataset limb --config ../config.yaml
bash 1_data_model_preparation/prepare_sccello.sh --dataset limb --config ../config.yaml

# Repeat for other models and datasets
```

##### Stage 2: Embedding Extraction

Extract embeddings for each model. **Note:** Each model may require a different environment setup.

```bash
# === Switch to GeneFormer environment ===
conda activate geneformer-env
python 2_extract_embedding/extract_embedding_geneformer.py \
    --dataset limb --config ../config.yaml

# === Switch to UCE environment ===
conda activate uce-env
bash 2_extract_embedding/extract_embedding_UCE.sh \
    --dataset limb --config ../config.yaml

# Repeat for other models and datasets
```

##### Stage 3: Integration

```bash

python 3_integration/integrate.py \
    --dataset limb --model uce --config ../config.yaml

# === Switch to CellPLM environment ===
conda activate cellplm-env
python 3_integration/integrate.py \
    --dataset limb --model cellplm --config ../config.yaml

python 3_integration/integrate.py \
    --dataset limb --model harmony --config ../config.yaml

# Repeat for other models and datasets
```

##### Stage 4: Benchmarking

Evaluate embedding quality using scIB metrics:

```bash
# Benchmark all models on limb dataset
python 4_benchmark/benchmark.py \
    --dataset limb --model uce --config ../config.yaml

python 4_benchmark/benchmark.py \
    --dataset limb --model geneformer --config ../config.yaml

# Repeat for other models and datasets
```

##### Stage 5: Batch Correction

Apply batch-centering correction to embeddings:

```bash
# Batch correct all models on limb dataset
python 5_batch_correction/batch_normalize.py \
    --dataset limb --model uce --config ../config.yaml

python 5_batch_correction/batch_normalize.py \
    --dataset limb --model geneformer --config ../config.yaml

# Repeat for other models and datasets
```

##### Stage 6: Linear Probing Analysis

Evaluate embeddings on downstream classification tasks:

```bash
# Original embeddings
python 6_probing/probing_main.py \
    --dataset limb --model uce --config ../config.yaml

# Batch-corrected embeddings
python 6_probing/probing_main.py \
    --dataset limb --model uce --batch-center --config ../config.yaml

# Repeat for other models and datasets
```



