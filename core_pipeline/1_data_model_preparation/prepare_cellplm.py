#!/usr/bin/env python3
"""
Prepare CellPLM model input from data.h5ad

CellPLM requires:
- X: raw count matrix
- var_names: ensembl ID

Process: data.h5ad → counts layer to X → var_names_make_unique() → save
"""

import argparse
import os
import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_cellplm(dataset_dir):
    """
    Prepare data for CellPLM model.
    
    Input: data.h5ad (from 0a_data_aggregation)
    Output: {dataset_dir}/cellplm/adata.h5ad
    
    Steps:
    1. Load data.h5ad
    2. Copy counts layer to X
    3. Ensure var_names are unique (ensembl IDs are already in index)
    4. Save as h5ad
    """
    logger.info(f"Preparing CellPLM input from {dataset_dir}")
    
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
    
    # Ensure var_names are unique
    adata.var_names_make_unique()
    logger.info("Ensured var_names are unique")
    
    # Create output directory and save
    cellplm_dir = os.path.join(dataset_dir, 'cellplm')
    os.makedirs(cellplm_dir, exist_ok=True)
    
    output_file = os.path.join(cellplm_dir, 'adata.h5ad')
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved CellPLM input to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CellPLM model input')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    
    args = parser.parse_args()
    prepare_cellplm(args.dataset)
