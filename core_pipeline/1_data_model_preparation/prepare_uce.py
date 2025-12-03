#!/usr/bin/env python3
"""
Prepare UCE model input from data.h5ad

UCE requires:
- X: raw count matrix
- var_names: gene names (symbols)

This is the simplest preparation step.
Process: data.h5ad → counts layer to X → set var_names to feature_name → save
"""

import argparse
import os
import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_uce(dataset_dir):
    """
    Prepare data for UCE model.
    
    Input: data.h5ad (from 0a_data_aggregation)
    Output: {dataset_dir}/uce/adata.h5ad
    
    Steps:
    1. Load data.h5ad
    2. Copy counts layer to X
    3. Set var_names to feature_name (gene symbols)
    4. Save as h5ad
    """
    logger.info(f"Preparing UCE input from {dataset_dir}")
    
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
    
    # Set var_names to feature_name (gene symbols)
    if 'feature_name' in adata.var.columns:
        adata.var.index = adata.var["feature_name"].astype(str)
        adata.var.index.name = None
        logger.info("Set var_names to feature_name (gene symbols)")
    else:
        logger.warning("feature_name not found in var, keeping original var_names")
    
    # Create output directory and save
    uce_dir = os.path.join(dataset_dir, 'uce')
    os.makedirs(uce_dir, exist_ok=True)
    
    output_file = os.path.join(uce_dir, 'adata.h5ad')
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved UCE input to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare UCE model input')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    
    args = parser.parse_args()
    prepare_uce(args.dataset)
