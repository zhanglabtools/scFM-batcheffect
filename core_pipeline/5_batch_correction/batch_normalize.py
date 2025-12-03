#!/usr/bin/env python3
"""
Batch normalization script: Apply batch effect correction to embeddings.

Supports three batch correction methods:
1. Batch centering (by cells): Subtract batch-wise mean embedding
2. Batch centering (by celltypes): Subtract batch-wise celltype-mean embedding
3. PCA debiasing: Remove top principal components

Usage:
    python batch_normalize.py --dataset /path/to/dataset --model {model_name} --method {method}
    python batch_normalize.py --dataset /path/to/limb --model geneformer --method batch_cells
"""

import os
import sys
import time
import argparse
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


class BatchNormalizer:
    """Apply batch effect correction to embeddings."""
    
    def __init__(self, dataset_dir, result_dir=None):
        """
        Initialize normalizer.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad and config.yaml
            result_dir: Result directory containing Embeddings_{model}.h5ad
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.batch_key = None
        self.label_key = None
        self.load_config()
    
    def load_config(self):
        """Load batch_key and label_key from config.yaml."""
        config_file = os.path.join(self.dataset_dir, 'config.yaml')
        
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if isinstance(config, dict):
                    self.batch_key = config.get('batch_key', 'batch')
                    self.label_key = config.get('celltype_key', 'cell_type')
                    
                    logger.info(f"Loaded config: batch_key={self.batch_key}, label_key={self.label_key}")
            except Exception as e:
                logger.warning(f"Failed to load config.yaml: {e}")
        
        # Set defaults if not loaded from config
        if not self.batch_key:
            self.batch_key = 'batch'
        if not self.label_key:
            self.label_key = 'cell_type'
    
    def load_data(self, model_name):
        """Load integrated embedding data."""
        model_result_dir = os.path.join(self.result_dir, model_name)
        
        emb_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Embeddings file not found: {emb_file}")
        
        adata = sc.read_h5ad(emb_file)
        logger.info(f"Loaded {model_name} data: {adata.shape}")
        
        return adata, model_result_dir
    
    def center_by_batch_cells(self, adata, embedding_key='X_emb', 
                             max_cells_per_batch=10000, random_seed=42, normalize=True):
        """
        Center embeddings by subtracting batch-wise mean (all cells).
        
        Args:
            adata: AnnData object
            embedding_key: Key in obsm for embedding
            max_cells_per_batch: Max cells to use for computing mean per batch
            random_seed: Random seed for sampling
            normalize: Whether to apply L2 normalization after centering
        
        Returns:
            centered_embeddings: Batch-centered embeddings
        """
        logger.info("Applying batch centering by cells (batch-wise mean)...")
        
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
    
    def center_by_batch_celltypes(self, adata, embedding_key='X_emb', normalize=True):
        """
        Center embeddings by subtracting batch-wise celltype-mean embedding.
        
        Args:
            adata: AnnData object
            embedding_key: Key in obsm for embedding
            normalize: Whether to apply L2 normalization after centering
        
        Returns:
            centered_embeddings: Batch-centered embeddings
        """
        logger.info("Applying batch centering by celltypes (batch-wise celltype-mean)...")
        
        embeddings = adata.obsm[embedding_key].copy()
        centered_embeddings = embeddings.copy()
        
        # Process each batch
        for batch in adata.obs[self.batch_key].unique():
            batch_mask = adata.obs[self.batch_key] == batch
            batch_indices = np.where(batch_mask)[0]
            
            # Get celltype means for this batch
            batch_obs = adata.obs.loc[batch_mask]
            batch_celltype_means = []
            
            for celltype in batch_obs[self.label_key].unique():
                celltype_mask = batch_obs[self.label_key] == celltype
                celltype_indices = np.where(celltype_mask)[0] + batch_indices[0]  # Adjust indices
                celltype_indices = batch_indices[batch_obs[self.label_key] == celltype]
                
                # Compute celltype mean
                celltype_mean = embeddings[celltype_indices].mean(axis=0)
                batch_celltype_means.append(celltype_mean)
            
            # Compute batch-wide celltype mean
            batch_celltype_mean = np.array(batch_celltype_means).mean(axis=0)
            
            # Center all cells in batch
            centered_embeddings[batch_indices] = embeddings[batch_indices] - batch_celltype_mean
            
            logger.info(f"  Batch {batch}: Used {len(batch_celltype_means)} celltypes to compute mean for {len(batch_indices)} cells")
        
        # L2 normalization
        if normalize:
            from sklearn.preprocessing import normalize as sk_normalize
            centered_embeddings = sk_normalize(centered_embeddings, norm="l2")
            logger.info("  Applied L2 normalization")
        
        return centered_embeddings
    
    def pca_debias(self, adata, embedding_key='X_emb', n_components=1, normalize=True):
        """
        Remove top principal components from embeddings.
        
        Args:
            adata: AnnData object
            embedding_key: Key in obsm for embedding
            n_components: Number of top components to remove
            normalize: Whether to apply L2 normalization after debiasing
        
        Returns:
            debiased_embeddings: PCA-debiased embeddings
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize as sk_normalize
        
        logger.info(f"Applying PCA debiasing (removing top {n_components} components)...")
        
        embeddings = adata.obsm[embedding_key].copy()
        
        # Fit PCA
        pca = PCA(n_components=min(n_components + 1, embeddings.shape[1]))
        pca.fit(embeddings)
        
        # Project to first n_components and subtract
        principal_components = pca.components_[:n_components]
        projections = embeddings.dot(principal_components.T).dot(principal_components)
        debiased_embeddings = embeddings - projections
        
        logger.info(f"  Removed components explain variance: {pca.explained_variance_ratio_[:n_components]}")
        
        # L2 normalization
        if normalize:
            debiased_embeddings = sk_normalize(debiased_embeddings, norm="l2")
            logger.info("  Applied L2 normalization")
        
        return debiased_embeddings
    
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
                        fig = sc.pl.umap(adata, color=col, show=False, size=10)
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
            kbet_per_label=True,
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
    
    def normalize(self, model_name, method='batch_cells'):
        """
        Full batch normalization pipeline.
        
        Args:
            model_name: Name of model to normalize
            method: Normalization method ('batch_cells', 'batch_celltypes', 'pca_debias')
        """
        logger.info(f"{'='*60}")
        logger.info(f"Starting batch normalization for: {model_name.upper()}")
        logger.info(f"Method: {method}")
        logger.info(f"{'='*60}")
        
        try:
            # Load data
            adata, model_result_dir = self.load_data(model_name)
            
            # Apply batch correction
            if method == 'batch_cells':
                corrected_emb = self.center_by_batch_cells(adata, embedding_key='X_emb')
            elif method == 'batch_celltypes':
                corrected_emb = self.center_by_batch_celltypes(adata, embedding_key='X_emb')
            elif method == 'pca_debias':
                corrected_emb = self.pca_debias(adata, embedding_key='X_emb')
            else:
                raise ValueError(f"Unknown method: {method}")
            
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


def batch_normalize_model(dataset_dir, model_name, result_dir=None, method='batch_cells'):
    """
    Apply batch normalization to a specific model.
    
    Args:
        dataset_dir: Dataset directory containing data.h5ad and config.yaml
        model_name: Name of the model to normalize
        result_dir: Result directory (default: {dataset_dir}/../Result/{dataset_name})
        method: Batch correction method ('batch_cells', 'batch_celltypes', 'pca_debias')
    """
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Infer result directory if not provided
    if result_dir is None:
        dataset_name = os.path.basename(dataset_dir)
        result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name)
    
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Result directory: {result_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    
    # Create normalizer and run
    try:
        normalizer = BatchNormalizer(dataset_dir, result_dir)
        normalizer.normalize(model_name, method)
        
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
Methods:
    batch_cells       - Subtract batch-wise mean embedding (all cells)
    batch_celltypes   - Subtract batch-wise celltype-mean embedding
    pca_debias        - Remove top principal components

Examples:
    python batch_normalize.py --dataset /path/to/limb --model geneformer --method batch_cells
    python batch_normalize.py --dataset /path/to/liver --model harmony --method pca_debias
    python batch_normalize.py --dataset /path/to/immune --model uce --method batch_celltypes --result-dir /custom/path

Supported Models:
    - uce, cellplm, geneformer, genecompass, scfoundation, sccello, nicheformer, scgpt
    - pca, scvi, harmony, scanorama
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset directory containing data.h5ad and config.yaml'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model or algorithm name to batch normalize'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='batch_cells',
        choices=['batch_cells', 'batch_celltypes', 'pca_debias'],
        help='Batch correction method'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='Result directory (default: inferred from dataset path)'
    )
    
    args = parser.parse_args()
    
    batch_normalize_model(args.dataset, args.model, args.result_dir, args.method)
