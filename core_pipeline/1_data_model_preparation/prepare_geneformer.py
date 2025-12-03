#!/usr/bin/env python3
"""
Prepare GeneFormer model input from data.h5ad

GeneFormer requires:
- X: raw count matrix
- var: ensembl_id column
- obs: n_counts (total counts per cell)

Process: data.h5ad → counts to X → add ensembl_id col → rename total_counts → save
Note: Actual tokenization happens in embedding extraction phase
"""

import argparse
import os
import scanpy as sc
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_geneformer(dataset_dir):
    """
    Prepare data for GeneFormer model.
    
    Input: data.h5ad (from 0a_data_aggregation)
    Output: {dataset_dir}/geneformer/adata.h5ad
    
    Steps:
    1. Load data.h5ad
    2. Copy counts layer to X
    3. Add ensembl_id to var (copy from index)
    4. Rename total_counts → n_counts in obs
    5. Save as h5ad
    """
    logger.info(f"Preparing GeneFormer input from {dataset_dir}")
    
    data_h5ad = os.path.join(dataset_dir, 'data.h5ad')
    if not os.path.exists(data_h5ad):
        raise FileNotFoundError(f"data.h5ad not found at {data_h5ad}")
    
    # Load data
    adata = sc.read_h5ad(data_h5ad)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Copy counts layer to X
    adata.X = adata.layers['counts'].copy()
    del adata.layers['counts']
    logger.info("Copied counts layer to X")
    
    # Add ensembl_id to var (if not already present)
    if 'ensembl_id' not in adata.var.columns:
        # Use index as ensembl_id if available, otherwise use gene names
        adata.var['ensembl_id'] = adata.var.index
        logger.info("Added ensembl_id column to var")
    
    # Rename total_counts → n_counts
    if 'total_counts' in adata.obs.columns:
        adata.obs['n_counts'] = adata.obs['total_counts']
        del adata.obs['total_counts']
        logger.info("Renamed total_counts → n_counts")
    elif 'n_counts' not in adata.obs.columns:
        # Calculate if not present
        adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        logger.info("Calculated n_counts from X")
    
    # Create output directory and save
    geneformer_dir = os.path.join(dataset_dir, 'geneformer')
    os.makedirs(geneformer_dir, exist_ok=True)
    
    output_file = os.path.join(geneformer_dir, 'adata.h5ad')
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved GeneFormer input to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GeneFormer model input')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    
    args = parser.parse_args()
    prepare_geneformer(args.dataset)
