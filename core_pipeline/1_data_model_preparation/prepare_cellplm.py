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
import yaml
import scanpy as sc
import logging

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


def prepare_cellplm(dataset_config):
    """
    Prepare data for CellPLM model.
    
    Input: data.h5ad (from data_preprocess.py)
    Output: {output_data_dir}/cellplm/adata.h5ad
    
    Steps:
    1. Load data.h5ad from output_data_dir
    2. Copy counts layer to X
    3. Ensure var_names are unique (ensembl IDs are already in index)
    4. Save to cellplm subdirectory
    """
    logger.info("Preparing CellPLM input...")
    
    # Get output directory from config
    output_dir = dataset_config['output_data_dir']
    data_h5ad = os.path.join(output_dir, 'data.h5ad')
    
    if not os.path.exists(data_h5ad):
        raise FileNotFoundError(f"data.h5ad not found at {data_h5ad}")
    
    # Load data
    adata = sc.read_h5ad(data_h5ad)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Copy counts layer to X
    if 'counts' not in adata.layers:
        raise ValueError("'counts' layer not found in data.h5ad. Make sure data_preprocess.py was run first.")
    
    adata.X = adata.layers['counts'].copy()
    del adata.layers['counts']
    logger.info("Copied counts layer to X")
    
    # Ensure var_names are unique
    adata.var_names_make_unique()
    logger.info("Ensured var_names are unique")
    
    # Create output directory and save
    cellplm_dir = os.path.join(output_dir, 'cellplm')
    os.makedirs(cellplm_dir, exist_ok=True)
    
    output_file = os.path.join(cellplm_dir, 'adata.h5ad')
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved CellPLM input to {output_file}")
    logger.info(f"Final shape: {adata.shape}")


def main():
    parser = argparse.ArgumentParser(description='Prepare CellPLM model input')
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
    logger.info(f"Loading dataset configuration for: {args.dataset}")
    
    # Prepare CellPLM input
    prepare_cellplm(dataset_config)


if __name__ == '__main__':
    main()