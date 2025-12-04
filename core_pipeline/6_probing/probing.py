#!/usr/bin/env python3
"""
Probing analysis script: Evaluate embeddings on downstream classification tasks

Supports both original and batch-corrected embeddings with 5-fold cross-validation

Usage:
    python probing_main.py --dataset limb --model geneformer --config config.yaml
    python probing_main.py --dataset limb --model geneformer --batch-center --config config.yaml
"""

import argparse
import os
import sys
import yaml
import logging
import time
from datetime import datetime
import warnings
import gc

import numpy as np
import pandas as pd
import scanpy as sc
from utils import CVDataSplitManager, process_single_fold_model

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def summarize_and_save_results(all_results, results_path, n_splits):
    """
    Summarize and save CV results
    
    Parameters:
        all_results (list): List of results from all tasks
        results_path (str): Path to save results
        n_splits (int): Number of splits
    """
    # Organize result data
    cv_data = []
    
    for result in all_results:
        if result['status'] == 'success':
            model_name = result['model_name']
            fold_idx = result['fold_idx']
            metrics = result['metrics']
            
            # Add overall accuracy and macro-f1
            cv_data.append({
                'model_name': model_name,
                'fold': fold_idx,
                'dataset': 'overall',
                'accuracy': metrics['overall_accuracy'],
                'f1_score': metrics['macro_f1'],
                'support': sum([m['support'] for m in metrics['class_metrics'].values()])
            })
            
            # Add per-class accuracy and f1
            for class_name, class_metric in metrics['class_metrics'].items():
                cv_data.append({
                    'model_name': model_name,
                    'fold': fold_idx,
                    'dataset': class_name,
                    'accuracy': class_metric['accuracy'],
                    'f1_score': class_metric['f1_score'],
                    'support': class_metric['support']
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(cv_data)
    
    if len(df) > 0:
        # Calculate statistical summary
        summary_data = []
        
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            
            for dataset in model_df['dataset'].unique():
                dataset_df = model_df[model_df['dataset'] == dataset]
                
                if len(dataset_df) >= n_splits:
                    # Calculate mean
                    summary_data.append({
                        'model_name': model_name,
                        'fold': 'mean',
                        'dataset': dataset,
                        'accuracy': dataset_df['accuracy'].mean(),
                        'f1_score': dataset_df['f1_score'].mean(),
                        'support': int(dataset_df['support'].mean())
                    })
                    
                    # Calculate standard deviation
                    summary_data.append({
                        'model_name': model_name,
                        'fold': 'std',
                        'dataset': dataset,
                        'accuracy': dataset_df['accuracy'].std(),
                        'f1_score': dataset_df['f1_score'].std(),
                        'support': int(dataset_df['support'].std())
                    })
        
        # Add summary data
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # Sort
        final_df = final_df.sort_values(['model_name', 'dataset', 'fold']).reset_index(drop=True)
        
        # Save results
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        final_df.to_csv(results_path, index=False, float_format='%.6f')
        
        logger.info(f"✓ CV results saved to: {results_path}")
        
        # Print brief summary
        overall_mean = final_df[(final_df['dataset'] == 'overall') & (final_df['fold'] == 'mean')]
        if len(overall_mean) > 0:
            logger.info("\nOverall Accuracy (mean ± std):")
            for _, row in overall_mean.iterrows():
                model_name = row['model_name']
                mean_acc = row['accuracy']
                std_row = final_df[(final_df['model_name'] == model_name) & 
                                  (final_df['dataset'] == 'overall') & 
                                  (final_df['fold'] == 'std')]
                if len(std_row) > 0:
                    std_acc = std_row.iloc[0]['accuracy']
                    logger.info(f"  {model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        logger.warning("No successful results to summarize")


def probing_main(dataset_name, model_name, config_path, batch_center=False):
    """
    Main probing analysis pipeline.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name to evaluate
        config_path: Path to config.yaml
        batch_center: Whether to use batch-corrected embeddings
    """
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get dataset config
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config")
    
    dataset_config = config['datasets'][dataset_name]
    result_dir = dataset_config.get('output_res_dir',
                                    os.path.join(dataset_config['output_data_dir'], 'Result', dataset_name))
    
    # Read batch_key and label_key from config
    batch_key = dataset_config.get('batch_key', 'batch')
    label_key = dataset_config.get('celltype_key', 'cell_type')
    
    # Read probing config parameters
    probing_config = config.get('probing', {})
    n_splits = probing_config.get('n_splits', 5)
    max_workers = probing_config.get('max_workers', 4)
    
    # Determine embedding file and type
    model_result_dir = os.path.join(result_dir, model_name)
    if batch_center:
        embedding_file = os.path.join(model_result_dir, f'Embeddings_{model_name}_batch_corrected.h5ad')
        embedding_key = 'X_emb_corrected'
        probing_type = 'batch_corrected'
    else:
        embedding_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        embedding_key = 'X_emb'
        probing_type = 'original'
    
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embeddings not found: {embedding_file}")
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Type: {probing_type}")
    logger.info(f"Batch key: {batch_key}")
    logger.info(f"Label key: {label_key}")
    logger.info(f"Embedding key: {embedding_key}")
    
    # Load data to validate
    logger.info("Loading data for validation...")
    adata = sc.read_h5ad(embedding_file)
    logger.info(f"Data shape: {adata.shape}")
    
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in obs")
    if label_key not in adata.obs.columns:
        raise ValueError(f"Label key '{label_key}' not found in obs")
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding key '{embedding_key}' not found in obsm")
    
    # Setup CV splits
    probing_dir = os.path.join(model_result_dir, f'probing_{probing_type}')
    splits_dir = os.path.join(probing_dir, 'cv_splits')
    
    cv_manager = CVDataSplitManager(splits_dir)
    
    try:
        cv_manager.load_cv_splits()
        logger.info("Loaded existing CV splits")
    except FileNotFoundError:
        logger.info("Creating new CV splits...")
        cv_manager.create_cv_splits(
            adata=adata,
            target_key=label_key,
            n_splits=n_splits,
            random_state=42,
            stratify_column=label_key
        )
    
    del adata
    gc.collect()
    
    # Prepare model config for utils_cv
    model_config = {
        'name': model_name,
        'adata_path': embedding_file,
        'embedding_key': embedding_key,
        'target_key': label_key,
        'save_path': probing_dir
    }
    
    # Run CV probing
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {n_splits}-fold CV probing")
    logger.info(f"{'='*60}")
    
    os.makedirs(probing_dir, exist_ok=True)
    
    all_results = []
    for fold_idx in range(n_splits):
        task = (model_config, fold_idx, cv_manager, False)
        result = process_single_fold_model(task)
        all_results.append(result)
        
        status = "✅" if result['status'] == 'success' else "❌"
        logger.info(f"{status} Fold {fold_idx + 1}/{n_splits}")
    
    # Save results
    results_file = os.path.join(probing_dir, 'probing_results.csv')
    summarize_and_save_results(all_results, results_file, n_splits)
    
    logger.info(f"✓ Probing analysis complete for {model_name}")
    logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Probing analysis: Evaluate embeddings on downstream classification tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python probing_main.py --dataset limb --model geneformer --config config.yaml
    python probing_main.py --dataset limb --model geneformer --batch-center --config config.yaml
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
        help='Dataset name'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model or algorithm name to evaluate'
    )
    parser.add_argument(
        '--batch-center',
        action='store_true',
        help='Use batch-corrected embeddings (default: original)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml file'
    )
    
    args = parser.parse_args()
    
    probing_main(args.dataset, args.model, args.config, batch_center=args.batch_center)
