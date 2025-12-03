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
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add config loader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config_loader import load_config, get_model_path, get_model_config

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.warning(f"Could not load config.yaml, using hardcoded defaults: {e}")
    config = None

# scFoundation gene list path (from config or hardcoded default)
def get_scfoundation_gene_list():
    if config:
        return get_model_path(config, 'scfoundation', 'gene_list',
               "/home/wanglinting/LCBERT/Code/model-scFoundation/OS_scRNA_gene_index.19264.tsv")
    return "/home/wanglinting/LCBERT/Code/model-scFoundation/OS_scRNA_gene_index.19264.tsv"

SCFOUNDATION_GENE_LIST = get_scfoundation_gene_list()


def prepare_scfoundation(dataset_dir):
    """
    Prepare data for scFoundation model.
    
    Input: data.h5ad (from 0a_data_aggregation, X is log1p-normalized)
    Output: {dataset_dir}/scfoundation/data.npz (sparse matrix, 19264 genes)
    
    Steps:
    1. Load data.h5ad
    2. Read 19264-gene list
    3. Select/pad genes to match exact list
    4. Save sparse matrix as NPZ
    """
    logger.info(f"Preparing scFoundation input from {dataset_dir}")
    
    # Check input file
    data_h5ad = os.path.join(dataset_dir, 'data.h5ad')
    if not os.path.exists(data_h5ad):
        raise FileNotFoundError(f"data.h5ad not found at {data_h5ad}")
    
    # Check gene list file
    if not os.path.exists(SCFOUNDATION_GENE_LIST):
        raise FileNotFoundError(f"Gene list not found at {SCFOUNDATION_GENE_LIST}")
    
    # Load data
    adata = sc.read_h5ad(data_h5ad)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Read gene list (19264 genes)
    gene_df = pd.read_csv(SCFOUNDATION_GENE_LIST, header=0, delimiter='\t')
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
    scfoundation_dir = os.path.join(dataset_dir, 'scfoundation')
    os.makedirs(scfoundation_dir, exist_ok=True)
    
    output_file = os.path.join(scfoundation_dir, 'data.npz')
    sp.save_npz(output_file, X_selected)
    logger.info(f"Saved scFoundation input to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare scFoundation model input')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    
    args = parser.parse_args()
    prepare_scfoundation(args.dataset)
    X_sparse = sp.csr_matrix(adata.X) if not sp.issparse(adata.X) else adata.X.tocsr()
    
    # Save sparse matrix
    npz_file = os.path.join(scfound_dir, 'data.npz')
    sp.save_npz(npz_file, X_sparse)
    logger.info(f"Saved sparse matrix to {npz_file}")
    
    # Save gene index mapping
    gene_index_file = os.path.join(scfound_dir, 'gene_names.json')
    gene_mapping = {i: gene for i, gene in enumerate(adata.var_names)}
    with open(gene_index_file, 'w') as f:
        json.dump(gene_mapping, f)
    logger.info(f"Saved gene mapping to {gene_index_file}")
    
    # Save cell metadata
    metadata_file = os.path.join(scfound_dir, 'cell_metadata.csv')
    adata.obs.to_csv(metadata_file)
    logger.info(f"Saved cell metadata to {metadata_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare scFoundation input')
    parser.add_argument('--data', type=str, required=True, help='Path to data.h5ad')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    prepare_scfoundation(args.data, args.output)
