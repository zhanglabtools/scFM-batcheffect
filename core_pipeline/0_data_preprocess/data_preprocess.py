#!/usr/bin/env python3
"""
Data aggregation and preprocessing script.
Converts raw CELLxGENE h5ad files to standardized data.h5ad format.

Datasets:
  - limb: Human and mouse limb tissues (cross-species alignment)
  - liver: Liver tissue (single-species, cell_type filtering)
  - Immune: Immune cells from donor D529 (subset of larger dataset)
  - HLCA_assay: HLCA human lung assay (filtered by study and assay type)
  - HLCA_disease: HLCA human lung disease (filtered by study and disease)
  - HLCA_sn: HLCA human lung single-nucleus (filtered by suspension type)
"""

import argparse
import os
import yaml
import pandas as pd
import scanpy as sc
import scib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_limb(config):
    """
    Preprocess limb dataset: human + mouse cross-species alignment.
    """
    logger.info("Preprocessing limb dataset...")
    
    # Read human and mouse data
    adata_human = sc.read_h5ad(config['human_path'])
    adata_mouse = sc.read_h5ad(config['mouse_path'])
    logger.info(f"Human data: {adata_human.shape}, Mouse data: {adata_mouse.shape}")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata_human.X = None
    adata_human.X = adata_human.raw.X
    adata_human.raw = None
    
    adata_mouse.X = None
    adata_mouse.X = adata_mouse.raw.X
    adata_mouse.raw = None
    
    # Filter mouse data by sequencing center
    adata_mouse = adata_mouse[adata_mouse.obs["sequencing_center"] == "Sanger"]
    logger.info(f"After filtering: Mouse data {adata_mouse.shape}")
    
    # Load homology mapping
    homology = pd.read_csv(config['homology_path'])
    homology = homology.rename(columns={
        "Gene stable ID": "human_ensembl",
        "Mouse gene stable ID": "mouse_ensembl",
        "Gene name": "human_symbol",
        "Mouse gene name": "mouse_symbol"
    })
    homology = homology.dropna(subset=["human_ensembl", "mouse_ensembl"])
    homology = homology.drop_duplicates(subset=["human_ensembl"])
    logger.info(f"Homolog pairs: {len(homology)}")
    
    # Extract common genes
    human_common = adata_human[:, adata_human.var_names.isin(homology["human_ensembl"])].copy()
    mouse_common = adata_mouse[:, adata_mouse.var_names.isin(homology["mouse_ensembl"])].copy()
    
    # Map mouse genes to human IDs
    mouse2human = dict(zip(homology["mouse_ensembl"], homology["human_ensembl"]))
    mouse_common.var["human_ensembl"] = mouse_common.var_names.map(mouse2human)
    mouse_common.var_names = mouse_common.var["human_ensembl"].values
    mouse_common.var_names_make_unique()
    
    # Align gene order: take intersection and sort
    common_genes = sorted(human_common.var_names.intersection(mouse_common.var_names))
    mouse_common = mouse_common[:, common_genes]
    human_common = human_common[:, common_genes]
    
    # Add organism label and align obs columns
    human_common.obs["organism"] = "human"
    mouse_common.obs["organism"] = "mouse"
    
    obs_cols = list(set(human_common.obs.columns) | set(mouse_common.obs.columns))
    human_common.obs = human_common.obs.reindex(columns=obs_cols)
    mouse_common.obs = mouse_common.obs.reindex(columns=obs_cols)
    
    # Concatenate using join='inner' to keep common genes
    adata = human_common.concatenate(
        mouse_common,
        join="inner",
        batch_key=None
    )
    logger.info(f"Concatenated data: {adata.shape}")
    
    # Restore feature_name from human data
    adata.var["feature_name"] = adata_human.var.loc[adata.var_names, "feature_name"]
    
    # Clear unnecessary structures
    adata.obsm.clear()
    
    return adata


def preprocess_liver(config):
    """
    Preprocess liver dataset: filter unknown cell types.
    """
    logger.info("Preprocessing liver dataset...")
    
    adata = sc.read_h5ad(config['data_path'])
    logger.info(f"Raw data: {adata.shape}")
    
    # Filter unknown cell types
    adata = adata[adata.obs['cell_type'] != 'unknown', :].copy()
    logger.info(f"After filtering unknown: {adata.shape}")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata.X = None
    adata.X = adata.raw.X
    adata.raw = None
    
    # Clear unnecessary structures
    adata.uns.clear()
    adata.obsm.clear()
    
    return adata


def preprocess_immune(config):
    """
    Preprocess immune dataset: extract single donor D529 and create cell type annotations.
    """
    logger.info("Preprocessing immune dataset...")
    
    adata_raw = sc.read_h5ad(config['data_path'])
    logger.info(f"Raw data: {adata_raw.shape}")
    
    # Filter by donor D529
    adata = adata_raw[adata_raw.obs['donor_id'] == 'D529'].copy()
    logger.info(f"After filtering donor D529: {adata.shape}")
    
    # Create detailed cell type mappings
    # Level 2: detailed cell types
    mapping_level_2 = {
        # Monocytes & Macrophages
        "macrophage": "Macrophages",
        "classical monocyte": "Monocytes",
        "non-classical monocyte": "Monocytes",
        # NK cells
        "CD16-positive, CD56-dim natural killer cell, human": "NK cells",
        "CD16-negative, CD56-bright natural killer cell, human": "NK cells",
        # CD4+ T cells
        "central memory CD4-positive, alpha-beta T cell": "CD4+ T cells",
        "effector memory CD4-positive, alpha-beta T cell": "CD4+ T cells",
        "effector memory CD4-positive, alpha-beta T cell, terminally differentiated": "CD4+ T cells",
        "CD4-positive, alpha-beta memory T cell": "CD4+ T cells",
        "naive thymus-derived CD4-positive, alpha-beta T cell": "CD4+ T cells",
        "CD4-positive, CD25-positive, alpha-beta regulatory T cell": "CD4+ T cells",
        # CD8+ T cells
        "effector memory CD8-positive, alpha-beta T cell": "CD8+ T cells",
        "effector memory CD8-positive, alpha-beta T cell, terminally differentiated": "CD8+ T cells",
        "CD8-positive, alpha-beta memory T cell": "CD8+ T cells",
        "naive thymus-derived CD8-positive, alpha-beta T cell": "CD8+ T cells",
        "central memory CD8-positive, alpha-beta T cell": "CD8+ T cells",
        # Unconventional T cells
        "gamma-delta T cell": "Unconventional T cells",
        "mucosal invariant T cell": "Unconventional T cells",
        # B lineage
        "naive B cell": "B cells",
        "memory B cell": "B cells",
        "germinal center B cell": "B cells",
        "plasmablast": "B cells",
        "plasma cell": "B cells",
        "B cell": "B cells",
        # Innate lymphoid cells
        "immature innate lymphoid cell": "Innate lymphoid cells",
        "group 1 innate lymphoid cell": "Innate lymphoid cells",
        "group 3 innate lymphoid cell": "Innate lymphoid cells",
        # Dendritic cells
        "conventional dendritic cell": "Dendritic cells",
        "plasmacytoid dendritic cell": "Dendritic cells",
        # Other
        "mast cell": "Mast cells",
        "progenitor cell": "Progenitor cells",
    }
    
    adata.obs["cell_type_level_2"] = adata.obs["cell_type"].map(mapping_level_2)
    
    # Level 1: gross cell type categories
    mapping_level_1 = {
        "t_cell": "T cells",
        "b_cell": "B cells",
        "myeloid": "Myeloid cells",
        "nk_ilc": "NK/ILCs",
        "mast_cell": "Mast cells",
        "progenitor": "Progenitor cells"
    }
    
    adata.obs["cell_type_level_1"] = adata.obs["Gross_annotation"].map(mapping_level_1)
    logger.info("Cell type annotations created")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata.X = None
    adata.X = adata.raw.X
    adata.raw = None
    
    # Clear unnecessary structures
    adata.obsm.clear()
    adata.uns.clear()
    
    return adata


def preprocess_hlca_assay(config):
    """
    Preprocess HLCA assay dataset: filter by study.
    """
    logger.info("Preprocessing HLCA assay dataset...")
    
    adata_raw = sc.read_h5ad(config['data_path'])
    logger.info(f"Raw data: {adata_raw.shape}")
    
    # Filter by studies
    studies = config.get('studies', ["Sun_2020", "Meyer_2019"])
    adata = adata_raw[adata_raw.obs['study'].isin(studies)].copy()
    logger.info(f"After filtering studies: {adata.shape}")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata.X = None
    adata.X = adata.raw.X
    adata.raw = None
    
    # Clear unnecessary structures
    adata.uns.clear()
    adata.obsm.clear()
    
    return adata


def preprocess_hlca_disease(config):
    """
    Preprocess HLCA disease dataset: filter by study.
    """
    logger.info("Preprocessing HLCA disease dataset...")
    
    adata_raw = sc.read_h5ad(config['data_path'])
    logger.info(f"Raw data: {adata_raw.shape}")
    
    # Filter by studies
    studies = config.get('studies', ["Sun_2020", "Meyer_2019"])
    adata = adata_raw[adata_raw.obs['study'].isin(studies)].copy()
    logger.info(f"After filtering studies: {adata.shape}")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata.X = None
    adata.X = adata.raw.X
    adata.raw = None
    
    # Clear unnecessary structures
    adata.uns.clear()
    adata.obsm.clear()
    
    return adata


def preprocess_hlca_sn(config):
    """
    Preprocess HLCA single-nucleus dataset: filter by study.
    """
    logger.info("Preprocessing HLCA single-nucleus dataset...")
    
    adata_raw = sc.read_h5ad(config['data_path'])
    logger.info(f"Raw data: {adata_raw.shape}")
    
    # Filter by studies
    studies = config.get('studies', ["Sun_2020", "Meyer_2019"])
    adata = adata_raw[adata_raw.obs['study'].isin(studies)].copy()
    logger.info(f"After filtering studies: {adata.shape}")
    
    # Set X to None, then extract raw counts from .raw attribute
    adata.X = None
    adata.X = adata.raw.X
    adata.raw = None
    
    # Clear unnecessary structures
    adata.uns.clear()
    adata.obsm.clear()
    
    return adata


def standardize_preprocessing(adata, config):
    """
    Apply standard preprocessing steps to all datasets.
    - Save raw counts to layers['counts']
    - Add cell_id
    - Calculate QC metrics
    - Normalize and log1p transform
    - Feature selection and PCA
    - Clear unnecessary obs columns
    """
    logger.info("Applying standardized preprocessing...")
    
    # Save raw counts to layer (if not already done)
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    
    # Add cell_id
    adata.obs['cell_id'] = [f"cell_{i}" for i in range(adata.n_obs)]
    
    # QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    logger.info(f"QC metrics calculated")
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.info(f"Data normalized and log-transformed")
    
    # Feature selection and dimensionality reduction
    batch_key = config.get('batch_key', 'batch')
    if batch_key in adata.obs.columns:
        scib.pp.reduce_data(
            adata, 
            n_top_genes=config.get('n_top_genes', 2000),
            batch_key=batch_key,
            pca=True,
            neighbors=False
        )
        logger.info(f"Feature selection and PCA done (batch-aware)")
    else:
        logger.warning(f"Batch key '{batch_key}' not found in obs, using non-batch-aware feature selection")
        sc.pp.highly_variable_genes(adata, n_top_genes=config.get('n_top_genes', 2000))
        adata.X = adata.X[:, adata.var['highly_variable']]
        sc.pp.pca(adata, n_comps=50)
        logger.info(f"Feature selection and PCA done")
    
    # Drop unnecessary columns
    cols_to_drop = [col for col in adata.obs.columns if col in ['leiden', 'leiden_R']]
    if cols_to_drop:
        adata.obs.drop(cols_to_drop, axis=1, inplace=True)
        logger.info(f"Dropped unnecessary obs columns: {cols_to_drop}")
    
    return adata


def main():
    parser = argparse.ArgumentParser(description='Aggregate and preprocess scRNA-seq data')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    parser.add_argument('--output', type=str, default='data.h5ad',
                        help='Output file path (default: data.h5ad)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Dataset-specific preprocessing
    if args.dataset == 'limb':
        adata = preprocess_limb(config)
    elif args.dataset == 'liver':
        adata = preprocess_liver(config)
    elif args.dataset == 'Immune':
        adata = preprocess_immune(config)
    elif args.dataset == 'HLCA_assay':
        adata = preprocess_hlca_assay(config)
    elif args.dataset == 'HLCA_disease':
        adata = preprocess_hlca_disease(config)
    elif args.dataset == 'HLCA_sn':
        adata = preprocess_hlca_sn(config)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Standardized preprocessing
    adata = standardize_preprocessing(adata, config)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    adata.write_h5ad(args.output, compression='gzip')
    logger.info(f"Saved preprocessed data to {args.output}")
    logger.info(f"Final shape: {adata.shape}")


if __name__ == '__main__':
    main()
