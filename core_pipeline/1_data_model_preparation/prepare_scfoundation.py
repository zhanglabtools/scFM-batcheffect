#!/usr/bin/env python3
"""
Prepare scFoundation model input from data.h5ad

scFoundation requires:
- 19264-gene list subset of log1p-normalized expression
- Output as sparse NPZ matrix

Process:
1. Load data.h5ad (X should be log1p-normalized)
2. Read gene list from OS_scRNA_gene_index.19264.tsv (19264 genes)
3. Subset/pad to exactly 19264 genes in specific order
4. Save as sparse NPZ matrix
"""

import argparse
import os
import logging
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

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


def prepare_scfoundation(dataset_config, model_config):
    """
    Prepare data for scFoundation model.
    
    Input: data.h5ad (from data_preprocess.py, X is log1p-normalized)
    Output: {output_data_dir}/scfoundation/data.npz (sparse matrix, 19264 genes)
    
    Steps:
    1. Load data.h5ad
    2. Read 19264-gene list
    3. Select/pad genes to match exact list
    4. Save sparse matrix as NPZ
    """
    logger.info("Preparing scFoundation input...")
    
    # Get output directory from config
    output_dir = dataset_config['output_data_dir']
    data_h5ad = os.path.join(output_dir, 'data.h5ad')
    
    if not os.path.exists(data_h5ad):
        raise FileNotFoundError(f"data.h5ad not found at {data_h5ad}")
    
    # Get gene list path from model_paths config
    gene_list_path = model_config['scfoundation']['gene_list']
    
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list not found at {gene_list_path}")
    
    # Load data
    adata = sc.read_h5ad(data_h5ad)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Read gene list (19264 genes)
    gene_df = pd.read_csv(gene_list_path, header=0, delimiter='\t')
    gene_list = list(gene_df['gene_name'])
    logger.info(f"Loaded gene list with {len(gene_list)} genes")
    
    # Get current gene names
    adata_genes = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(adata_genes)}
    
    # Count genes to pad
    to_fill = [g for g in gene_list if g not in gene_to_idx]
    logger.info(f"Genes to pad with zeros: {len(to_fill)} / {len(gene_list)}")
    
    # Build sparse matrix by selecting/padding genes
    X = adata.X
    is_sparse = sp.issparse(X)
    
    selected_cols = []
    for g in gene_list:
        if g in gene_to_idx:
            col_idx = gene_to_idx[g]
            if is_sparse:
                col = X[:, col_idx]
            else:
                col = sp.csr_matrix(X[:, col_idx]).T
        else:
            # Zero column for missing gene
            col = sp.csr_matrix((adata.n_obs, 1))
        selected_cols.append(col)
    
    # Stack columns horizontally
    X_selected = sp.hstack(selected_cols, format='csr')
    logger.info(f"Built sparse matrix: {X_selected.shape}")
    
    # Create output directory and save
    scfoundation_dir = os.path.join(output_dir, 'scfoundation')
    os.makedirs(scfoundation_dir, exist_ok=True)
    
    output_file = os.path.join(scfoundation_dir, 'data.npz')
    sp.save_npz(output_file, X_selected)
    logger.info(f"Saved scFoundation input to {output_file}")
    logger.info(f"Final shape: {X_selected.shape}")


def main():
    parser = argparse.ArgumentParser(description='Prepare scFoundation model input')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract dataset-specific configuration
    dataset_config = config['datasets'][args.dataset]
    model_config = config['model_paths']
    logger.info(f"Loading dataset configuration for: {args.dataset}")
    
    # Prepare scFoundation input
    prepare_scfoundation(dataset_config)


if __name__ == '__main__':
    main()
