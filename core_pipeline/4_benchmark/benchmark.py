#!/usr/bin/env python3
"""
Benchmark script: Evaluate embedding quality using scIB metrics.
Computes biological conservation and batch correction metrics for integrated embeddings.

Usage:
    python benchmark.py --dataset /path/to/dataset --model {model_name}
    python benchmark.py --dataset /path/to/dataset --result-dir /path/to/result --model geneformer
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


class BenchmarkEvaluator:
    """Evaluate embedding quality using scIB metrics."""
    
    def __init__(self, dataset_dir, result_dir=None):
        """
        Initialize evaluator.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad and config.yaml
            result_dir: Result directory containing Embeddings_{model}.h5ad
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.batch_key = None
        self.label_key = None
        self.pre_integrated_embedding_key = None
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
                    self.pre_integrated_embedding_key = config.get('pre_integrated_embedding_obsm_key', 'X_pca')
                    
                    logger.info(f"Loaded config: batch_key={self.batch_key}, label_key={self.label_key}")
            except Exception as e:
                logger.warning(f"Failed to load config.yaml: {e}")
        
        # Set defaults if not loaded from config
        if not self.batch_key:
            self.batch_key = 'batch'
        if not self.label_key:
            self.label_key = 'cell_type'
        if not self.pre_integrated_embedding_key:
            self.pre_integrated_embedding_key = 'X_pca'
    
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
    
    def run_scib_benchmark(self, adata, model_name, model_result_dir):
        """Run comprehensive scIB benchmark."""
        try:
            from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
        except ImportError:
            logger.error("scib_metrics not installed. Install with: pip install scib-metrics")
            raise
        
        logger.info(f"Running scIB benchmark for {model_name}...")
        
        # Define metrics to compute
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
        
        # Run benchmarking (60% bio, 40% batch)
        benchmarker = Benchmarker(
            adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=["X_emb"],
            pre_integrated_embedding_obsm_key=self.pre_integrated_embedding_key,
            n_jobs=4,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batchcor,
        )
        
        benchmarker.benchmark()
        results_df = benchmarker.get_results(min_max_scale=False)
        
        # Save results
        results_file = os.path.join(model_result_dir, f'metrics_{model_name}.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"✓ Saved benchmark results to {results_file}")
        
        return results_df
    
    def benchmark(self, model_name):
        """Run full benchmark pipeline."""
        logger.info(f"{'='*60}")
        logger.info(f"Starting benchmark for: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load data
            adata, model_result_dir = self.load_data(model_name)
            
            # Validate metadata
            if not self.validate_metadata(adata):
                logger.error("Required metadata columns missing")
                return None
            
            # Run benchmark
            results_df = self.run_scib_benchmark(adata, model_name, model_result_dir)
            
            logger.info(f"✓ Benchmark complete for {model_name.upper()}")
            logger.info(f"{'='*60}\n")
            
            return results_df
        
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(f"{'='*60}\n")
            raise


def benchmark_model(dataset_dir, model_name, result_dir=None):
    """
    Run benchmark for a specific model.
    
    Args:
        dataset_dir: Dataset directory containing data.h5ad and config.yaml
        model_name: Name of the model to benchmark
        result_dir: Result directory (default: {dataset_dir}/../Result/{dataset_name})
    """
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Infer result directory if not provided
    if result_dir is None:
        dataset_name = os.path.basename(dataset_dir)
        result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name)
    
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Result directory: {result_dir}")
    logger.info(f"Model: {model_name}")
    
    # Create evaluator and run
    try:
        evaluator = BenchmarkEvaluator(dataset_dir, result_dir)
        results_df = evaluator.benchmark(model_name)
        
        if results_df is not None:
            logger.info("\nBenchmark Results:")
            print(results_df.to_string())
        
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        return True
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate embedding quality using scIB metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py --dataset /path/to/limb --model geneformer
    python benchmark.py --dataset /path/to/liver --model harmony --result-dir /custom/path
    python benchmark.py --dataset /path/to/immune --model pca

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
        help='Model or algorithm name to benchmark'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='Result directory (default: inferred from dataset path)'
    )
    
    args = parser.parse_args()
    
    benchmark_model(args.dataset, args.model, args.result_dir)
    
    config = {}
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    run_benchmark(args.dataset, config)
