#!/usr/bin/env python3
"""
Prepare GeneFormer model input from data.h5ad

GeneFormer requires:
- X: raw count matrix
- var: ensembl_id column
- obs: n_counts (total counts per cell)

Process: data.h5ad → counts to X → add ensembl_id col → rename total_counts → save
Then tokenize using GeneFormer's TranscriptomeTokenizer
"""

import argparse
import os
import sys
import yaml
import scanpy as sc
import numpy as np
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


def prepare_geneformer(dataset_config, model_config):
    """
    Prepare data for GeneFormer model.
    
    Input: data.h5ad (from data_preprocess.py)
    Output: {output_data_dir}/geneformer/adata.h5ad and tokenized files
    
    Steps:
    1. Load data.h5ad
    2. Copy counts layer to X
    3. Add ensembl_id to var (copy from index)
    4. Rename total_counts → n_counts in obs
    5. Save as h5ad
    6. Tokenize using GeneFormer TranscriptomeTokenizer
    """
    logger.info("Preparing GeneFormer input...")
    
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
    
    # Add ensembl_id to var (if not already present)
    if 'ensembl_id' not in adata.var.columns:
        adata.var['ensembl_id'] = adata.var.index
        logger.info("Added ensembl_id column to var")
    
    # Rename total_counts → n_counts
    if 'total_counts' in adata.obs.columns:
        adata.obs['n_counts'] = adata.obs['total_counts']
        del adata.obs['total_counts']
        logger.info("Renamed total_counts → n_counts")
    elif 'n_counts' not in adata.obs.columns:
        adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        logger.info("Calculated n_counts from X")
    
    # Create output directory and save
    geneformer_dir = os.path.join(output_dir, 'geneformer')
    os.makedirs(geneformer_dir, exist_ok=True)
    
    output_file = os.path.join(geneformer_dir, 'adata.h5ad')
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved GeneFormer input to {output_file}")
    logger.info(f"Final shape: {adata.shape}")
    
    # Get GeneFormer configuration
    gf_config = model_config.get('geneformer', {})
    module_path = gf_config.get('module_path', "/home/wanglinting/LCBERT/Code/model-geneformer")
    thread_num = gf_config.get('nproc', 4)
    
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"GeneFormer module path not found: {module_path}")
    
    logger.info(f"Using GeneFormer module path: {module_path}")
    logger.info(f"Using thread count: {thread_num}")
    
    # Import GeneFormer tokenizer
    sys.path.insert(0, module_path)
    try:
        from geneformer import TranscriptomeTokenizer
    except ImportError as e:
        raise ImportError(f"Failed to import TranscriptomeTokenizer from {module_path}: {e}")
    
    # Tokenize
    logger.info("Tokenizing with GeneFormer...")
    metadata_map = {'cell_id': 'cell_id'}
    
    tk = TranscriptomeTokenizer(metadata_map, nproc=thread_num)
    tk.tokenize_data(
        data_directory=geneformer_dir,
        output_directory=geneformer_dir,
        output_prefix="geneformer",
        file_format="h5ad"
    )
    logger.info("GeneFormer tokenization complete")


def main():
    parser = argparse.ArgumentParser(description='Prepare GeneFormer model input')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract dataset-specific configuration and model configuration
    dataset_config = config['datasets'][args.dataset]
    model_config = config['model_paths']
    logger.info(f"Loading dataset configuration for: {args.dataset}")
    
    # Prepare GeneFormer input
    prepare_geneformer(dataset_config, model_config)


if __name__ == '__main__':
    main()