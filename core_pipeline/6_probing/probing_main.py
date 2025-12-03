#!/usr/bin/env python3
"""
Probing analysis wrapper script
Evaluates embeddings on downstream classification tasks

Usage:
    python probing_main.py --dataset /path/to/dataset --model geneformer [--type original]
    python probing_main.py --dataset /path/to/dataset --model geneformer --type batch_normalized
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load config from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def probing_original(dataset_dir, model_name, result_dir):
    """
    Run probing on original (non-batch-corrected) embeddings using 5-fold cross-validation.
    
    Args:
        dataset_dir: Dataset directory containing data and metadata
        model_name: Model name (e.g., 'geneformer', 'cellplm')
        result_dir: Result directory containing embeddings
    """
    try:
        from utils_cv import CVDataSplitManager, process_single_fold_model
        import scanpy as sc
        import pandas as pd
        import numpy as np
        import concurrent.futures
        
        logger.info(f"Starting probing analysis for {model_name.upper()} (original embeddings)")
        
        # Paths
        dataset_name = os.path.basename(dataset_dir)
        model_result_dir = os.path.join(result_dir, model_name)
        emb_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Embeddings not found: {emb_file}")
        
        # Load data
        adata = sc.read_h5ad(emb_file)
        logger.info(f"Loaded data: {adata.shape}")
        
        # Prepare output directory
        probing_dir = os.path.join(model_result_dir, 'probing_original')
        os.makedirs(probing_dir, exist_ok=True)
        
        # Determine key columns from data
        # Try to infer label key (cell type annotation) and batch key
        label_key = None
        batch_key = None
        
        # Common cell type annotation column names
        for col_name in ['celltype', 'cell_type', 'cell_label', 'annotation', 'seurat_clusters']:
            if col_name in adata.obs.columns:
                label_key = col_name
                break
        
        # Common batch/donor column names
        for col_name in ['batch', 'batch_id', 'donor', 'donor_id', 'sample', 'sample_id']:
            if col_name in adata.obs.columns:
                batch_key = col_name
                break
        
        if label_key is None:
            logger.warning("Could not infer label key from data. Using 'celltype' if available.")
            label_key = 'celltype' if 'celltype' in adata.obs.columns else None
        
        if label_key is None:
            raise ValueError("No valid label key found in data. Required for classification tasks.")
        
        logger.info(f"Using label key: {label_key}")
        if batch_key:
            logger.info(f"Using batch key: {batch_key}")
        
        # Setup CV splits directory
        splits_dir = os.path.join(probing_dir, 'cv_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        cv_config = {
            'split_save_dir': splits_dir,
            'n_splits': 5,
            'max_workers': 4,
            'force_recompute': False
        }
        
        # Prepare model configuration
        model_config = {
            'name': model_name,
            'adata_path': emb_file,
            'embedding_key': 'X_emb',  # Default embedding key
            'target_key': label_key,
            'batch_key': batch_key,
            'save_path': probing_dir
        }
        
        # Load CV data split manager
        cv_split_manager = CVDataSplitManager(cv_config['split_save_dir'])
        
        # Create or load CV splits
        try:
            cv_split_manager.load_cv_splits()
            logger.info(f"Loaded existing CV splits from {cv_config['split_save_dir']}")
        except FileNotFoundError:
            logger.info(f"Creating new CV splits...")
            cv_split_manager.create_cv_splits(adata.obs[label_key].values, n_splits=5)
        
        # Run CV probing
        logger.info(f"Running 5-fold cross-validation probing for {model_name}...")
        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cv_config['max_workers']) as executor:
            futures = []
            for fold_idx in range(cv_config['n_splits']):
                future = executor.submit(process_single_fold_model, (model_config, fold_idx, cv_split_manager, False))
                futures.append((fold_idx, future))
            
            # Collect results
            for fold_idx, future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                    status = "✅" if result.get('status') == 'success' else "❌"
                    acc = result.get('accuracy', 0)
                    logger.info(f"{status} Fold {fold_idx + 1}: Accuracy = {acc:.4f}")
                except Exception as e:
                    logger.error(f"❌ Fold {fold_idx + 1} failed: {e}")
        
        logger.info(f"✓ Probing completed for {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Probing failed: {e}")
        raise


def probing_batch_normalized(dataset_dir, model_name, result_dir):
    """
    Run probing on batch-corrected embeddings using 5-fold cross-validation.
    
    Args:
        dataset_dir: Dataset directory containing data and metadata
        model_name: Model name (e.g., 'geneformer', 'cellplm')
        result_dir: Result directory containing batch-corrected embeddings
    """
    try:
        from utils_cv import CVDataSplitManager, process_single_fold_model
        import scanpy as sc
        import pandas as pd
        import numpy as np
        import concurrent.futures
        
        logger.info(f"Starting probing analysis for {model_name.upper()} (batch-corrected embeddings)")
        
        # Paths
        dataset_name = os.path.basename(dataset_dir)
        model_result_dir = os.path.join(result_dir, model_name)
        emb_file = os.path.join(model_result_dir, f'Embeddings_{model_name}_batch_corrected.h5ad')
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Batch-corrected embeddings not found: {emb_file}")
        
        # Load data
        adata = sc.read_h5ad(emb_file)
        logger.info(f"Loaded data: {adata.shape}")
        
        # Prepare output directory
        probing_dir = os.path.join(model_result_dir, 'probing_batch_corrected')
        os.makedirs(probing_dir, exist_ok=True)
        
        # Determine key columns from data
        # Try to infer label key (cell type annotation) and batch key
        label_key = None
        batch_key = None
        
        # Common cell type annotation column names
        for col_name in ['celltype', 'cell_type', 'cell_label', 'annotation', 'seurat_clusters']:
            if col_name in adata.obs.columns:
                label_key = col_name
                break
        
        # Common batch/donor column names
        for col_name in ['batch', 'batch_id', 'donor', 'donor_id', 'sample', 'sample_id']:
            if col_name in adata.obs.columns:
                batch_key = col_name
                break
        
        if label_key is None:
            logger.warning("Could not infer label key from data. Using 'celltype' if available.")
            label_key = 'celltype' if 'celltype' in adata.obs.columns else None
        
        if label_key is None:
            raise ValueError("No valid label key found in data. Required for classification tasks.")
        
        logger.info(f"Using label key: {label_key}")
        if batch_key:
            logger.info(f"Using batch key: {batch_key}")
        
        # Setup CV splits directory
        splits_dir = os.path.join(probing_dir, 'cv_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        cv_config = {
            'split_save_dir': splits_dir,
            'n_splits': 5,
            'max_workers': 4,
            'force_recompute': False
        }
        
        # Prepare model configuration
        model_config = {
            'name': model_name,
            'adata_path': emb_file,
            'embedding_key': 'X_emb_batch',  # Batch-corrected embedding key
            'target_key': label_key,
            'batch_key': batch_key,
            'save_path': probing_dir
        }
        
        # Load CV data split manager
        cv_split_manager = CVDataSplitManager(cv_config['split_save_dir'])
        
        # Create or load CV splits
        try:
            cv_split_manager.load_cv_splits()
            logger.info(f"Loaded existing CV splits from {cv_config['split_save_dir']}")
        except FileNotFoundError:
            logger.info(f"Creating new CV splits...")
            cv_split_manager.create_cv_splits(adata.obs[label_key].values, n_splits=5)
        
        # Run CV probing
        logger.info(f"Running 5-fold cross-validation probing on batch-corrected embeddings...")
        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cv_config['max_workers']) as executor:
            futures = []
            for fold_idx in range(cv_config['n_splits']):
                future = executor.submit(process_single_fold_model, (model_config, fold_idx, cv_split_manager, False))
                futures.append((fold_idx, future))
            
            # Collect results
            for fold_idx, future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                    status = "✅" if result.get('status') == 'success' else "❌"
                    acc = result.get('accuracy', 0)
                    logger.info(f"{status} Fold {fold_idx + 1}: Accuracy = {acc:.4f}")
                except Exception as e:
                    logger.error(f"❌ Fold {fold_idx + 1} failed: {e}")
        
        logger.info(f"✓ Probing completed for {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Probing failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run probing analysis on embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Types:
    original           - Evaluate original integrated embeddings
    batch_normalized   - Evaluate batch-corrected embeddings

Examples:
    python probing_main.py --dataset /path/to/limb --model geneformer
    python probing_main.py --dataset /path/to/limb --model geneformer --type batch_normalized
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset directory containing config.yaml and data'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model or algorithm name to evaluate'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='original',
        choices=['original', 'batch_normalized'],
        help='Type of embeddings to evaluate'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='Result directory (default: auto-inferred from dataset)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Infer result directory if not provided
    if args.result_dir is None:
        dataset_name = os.path.basename(args.dataset)
        args.result_dir = os.path.join(args.dataset, '..', 'Result', dataset_name)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Result: {args.result_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Type: {args.type}")
    
    try:
        if args.type == 'batch_normalized':
            probing_batch_normalized(args.dataset, args.model, args.result_dir)
        else:
            probing_original(args.dataset, args.model, args.result_dir)
        
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        return 0
        
    except Exception as e:
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        return 1


if __name__ == '__main__':
    sys.exit(main())
