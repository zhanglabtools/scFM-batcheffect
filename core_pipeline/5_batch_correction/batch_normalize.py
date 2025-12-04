#!/usr/bin/env python3
"""
Batch normalization script: Apply batch effect correction to embeddings.

Batch centering method: Subtract batch-wise mean embedding

Usage:
    python batch_normalize.py --dataset limb --model geneformer --config config.yaml
    python batch_normalize.py --dataset Immune --model harmony --config config.yaml
"""

import argparse
import os
import sys
import yaml
import time
import logging
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Set timezone
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

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


class BatchNormalizer:
    """Apply batch effect correction to embeddings."""
    
    def __init__(self, dataset_dir, result_dir, batch_key, label_key, batch_correction_config):
        """
        Initialize normalizer.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad
            result_dir: Result directory containing Embeddings_{model}.h5ad
            batch_key: Column name for batch annotation
            label_key: Column name for cell type annotation
            batch_correction_config: Batch correction configuration dict
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.batch_key = batch_key
        self.label_key = label_key
        self.batch_correction_config = batch_correction_config
    
    def load_data(self, model_name):
        """Load integrated embedding data."""
        model_result_dir = os.path.join(self.result_dir, model_name)
        
        emb_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Embeddings file not found: {emb_file}")
        
        adata = sc.read_h5ad(emb_file)
        logger.info(f"Loaded {model_name} data: {adata.shape}")
        
        return adata, model_result_dir
    
    def validate_metadata(self, adata):
        """Validate that required metadata columns exist."""
        missing_keys = []
        
        if self.batch_key not in adata.obs.columns:
            missing_keys.append(self.batch_key)
        if self.label_key not in adata.obs.columns:
            missing_keys.append(self.label_key)
        
        if missing_keys:
            logger.warning(f"Missing metadata columns: {missing_keys}")
            logger.warning(f"Available obs columns: {adata.obs.columns.tolist()}")
            return False
        
        return True
    
    def center_by_batch_cells(self, adata, embedding_key='X_emb'):
        """
        Center embeddings by subtracting batch-wise mean (all cells).
        
        Args:
            adata: AnnData object
            embedding_key: Key in obsm for embedding
        
        Returns:
            centered_embeddings: Batch-centered embeddings
        """
        logger.info("Applying batch centering (batch-wise mean)...")
        
        # Get parameters from config
        max_cells_per_batch = self.batch_correction_config.get('batch_cells', {}).get('max_cells_per_batch', 10000)
        random_seed = self.batch_correction_config.get('batch_cells', {}).get('random_seed', 42)
        normalize = self.batch_correction_config.get('batch_cells', {}).get('normalize', True)
        
        embeddings = adata.obsm[embedding_key].copy()
        centered_embeddings = embeddings.copy()
        
        # Process each batch
        for batch in adata.obs[self.batch_key].unique():
            batch_mask = adata.obs[self.batch_key] == batch
            batch_indices = np.where(batch_mask)[0]
            
            # Sample if too many cells
            if len(batch_indices) > max_cells_per_batch:
                logger.info(f"  Batch {batch}: {len(batch_indices)} cells exceeds limit {max_cells_per_batch}, sampling...")
                np.random.seed(random_seed)
                sampled_indices = np.random.choice(batch_indices, max_cells_per_batch, replace=False)
            else:
                sampled_indices = batch_indices
            
            # Compute batch mean
            batch_mean = embeddings[sampled_indices].mean(axis=0)
            
            # Center all cells in batch
            centered_embeddings[batch_indices] = embeddings[batch_indices] - batch_mean
            
            logger.info(f"  Batch {batch}: Used {len(sampled_indices)} cells to compute mean for {len(batch_indices)} cells")
        
        # L2 normalization
        if normalize:
            from sklearn.preprocessing import normalize as sk_normalize
            centered_embeddings = sk_normalize(centered_embeddings, norm="l2")
            logger.info("  Applied L2 normalization")
        
        return centered_embeddings
    
    def compute_umap_visualization(self, adata, embedding_key='X_emb', model_result_dir=None):
        """Compute UMAP and create visualizations."""
        logger.info("Computing UMAP and visualizations...")
        
        # Compute neighbors
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
        
        # Compute UMAP
        sc.tl.umap(adata)
        
        # Visualize batch and celltype
        if model_result_dir:
            os.makedirs(model_result_dir, exist_ok=True)
            
            for col in [self.batch_key, self.label_key]:
                if col in adata.obs.columns:
                    try:
                        sc.pl.umap(adata, color=col, show=False, size=10)
                        output_file = os.path.join(model_result_dir, f'umap_{col}.png')
                        plt.savefig(output_file, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(f"✓ Saved visualization: {col}")
                    except Exception as e:
                        logger.warning(f"Failed to visualize {col}: {e}")
    
    def benchmark_corrected_embedding(self, adata, model_name, model_result_dir):
        """Run scIB benchmark on corrected embedding."""
        try:
            from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
        except ImportError:
            logger.warning("scib_metrics not installed, skipping benchmark")
            return
        
        logger.info("Running scIB benchmark on batch-corrected embedding...")
        
        # Define metrics
        biocons = BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=True,
            clisi_knn=False
        )
        
        batchcor = BatchCorrection(
            bras=True,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=True,
            pcr_comparison=False
        )
        
        # Run benchmark
        benchmarker = Benchmarker(
            adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=["X_emb_corrected"],
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=4,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batchcor,
        )
        
        benchmarker.benchmark()
        results_df = benchmarker.get_results(min_max_scale=False)
        
        # Save results
        results_file = os.path.join(model_result_dir, f'metrics_batch_corrected.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"✓ Saved benchmark results to {results_file}")
        
        return results_df
    
    def normalize(self, model_name):
        """
        Full batch normalization pipeline.
        
        Args:
            model_name: Name of model to normalize
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting batch normalization for: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load data
            adata, model_result_dir = self.load_data(model_name)
            
            # Validate metadata
            if not self.validate_metadata(adata):
                logger.error("Required metadata columns missing")
                return False
            
            # Apply batch correction (batch_cells method)
            corrected_emb = self.center_by_batch_cells(adata, embedding_key='X_emb')
            
            # Store corrected embedding
            adata.obsm['X_emb_corrected'] = corrected_emb
            
            # Compute UMAP
            self.compute_umap_visualization(adata, embedding_key='X_emb_corrected', 
                                          model_result_dir=model_result_dir)
            
            # Save corrected data
            output_file = os.path.join(model_result_dir, f'Embeddings_{model_name}_batch_corrected.h5ad')
            adata.write_h5ad(output_file, compression='gzip')
            logger.info(f"✓ Saved corrected embeddings to {output_file}")
            
            # Run benchmark
            self.benchmark_corrected_embedding(adata, model_name, model_result_dir)
            
            # Cleanup
            gc.collect()
            del adata
            
            logger.info(f"✓ Batch normalization complete for {model_name.upper()}")
            logger.info(f"{'='*60}\n")
            
            return True
        
        except Exception as e:
            logger.error(f"Batch normalization failed: {e}")
            logger.error(f"{'='*60}\n")
            raise


def batch_normalize_model(dataset_name, model_name, config_path):
    """
    Apply batch normalization to a specific model.
    
    Args:
        dataset_name: Dataset name (e.g., 'limb', 'liver')
        model_name: Name of the model to normalize
        config_path: Path to config.yaml
    """
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get dataset config
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config")
    
    dataset_config = config['datasets'][dataset_name]
    dataset_dir = dataset_config['output_data_dir']
    result_dir = dataset_config.get('output_res_dir',
                                    os.path.join(dataset_dir, 'Result', dataset_name))
    batch_key = dataset_config.get('batch_key', 'batch')
    label_key = dataset_config.get('celltype_key', 'cell_type')
    batch_correction_config = config.get('batch_correction', {})
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Batch key: {batch_key}")
    logger.info(f"Label key: {label_key}")
    
    # Create normalizer and run
    try:
        normalizer = BatchNormalizer(dataset_dir, result_dir, batch_key, label_key, batch_correction_config)
        normalizer.normalize(model_name)
        
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        return True
    
    except Exception as e:
        logger.error(f"Batch normalization failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply batch effect correction to embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Batch Correction Method:
    batch_cells - Subtract batch-wise mean embedding (all cells)

Examples:
    python batch_normalize.py --dataset limb --model geneformer --config config.yaml
    python batch_normalize.py --dataset Immune --model harmony --config config.yaml
    python batch_normalize.py --dataset liver --model pca --config config.yaml

Supported Models:
    - uce, cellplm, geneformer, genecompass, scfoundation, sccello, nicheformer, scgpt
    - pca, scvi, harmony, scanorama
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
        help='Model or algorithm name to batch normalize'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.yaml file'
    )
    
    args = parser.parse_args()
    
    batch_normalize_model(args.dataset, args.model, args.config)