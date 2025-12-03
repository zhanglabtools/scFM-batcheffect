#!/usr/bin/env python3
"""
Integration script: Combine embeddings with original AnnData object

For each model, loads its embedding and:
1. Loads original standardized data (data.h5ad)
2. Loads embedding with model-specific logic
3. Integrates embedding into adata.obsm['X_emb']
4. Performs UMAP visualization on embedding space
5. Saves integrated data with visualizations

Supports 12 models/algorithms:
- Embedding models: UCE, CellPLM, GeneFormer, GeneCompass, scFoundation, scCello, NicheFormer
- Integration methods: scVI, Harmony, Scanorama, PCA

Usage:
    python integrate.py --dataset limb --model geneformer --config config.yaml
"""

import argparse
import os
import sys
import yaml
import glob
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


class EmbeddingIntegrator:
    """Integrate embeddings from different models."""
    
    def __init__(self, dataset_dir, result_dir, config):
        """
        Initialize integrator.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad
            result_dir: Result directory containing embeddings
            config: Configuration dictionary
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.config = config
        self.adata = None
    
    def load_data(self):
        """Load original standardized data."""
        data_file = os.path.join(self.dataset_dir, 'data.h5ad')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"data.h5ad not found at {data_file}")
        
        logger.info(f"Loading data from {data_file}")
        self.adata = sc.read_h5ad(data_file)
        logger.info(f"Data shape: {self.adata.shape}")
        
        return self.adata
    
    # ===========================
    # EMBEDDING MODEL LOADERS
    # ===========================
    
    def load_embedding_uce(self, model_result_dir):
        """Load UCE embedding from h5ad."""
        emb_files = glob.glob(os.path.join(model_result_dir, 'adata_uce_*.h5ad'))
        
        if not emb_files:
            raise FileNotFoundError(f"No UCE embedding h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(emb_files[0])
        embedding = adata_emb.obsm.get('X_uce', adata_emb.X)
        
        logger.info(f"Loaded UCE embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_cellplm(self, model_result_dir):
        """Load CellPLM embedding from h5ad."""
        emb_files = glob.glob(os.path.join(model_result_dir, '*.h5ad'))
        
        if not emb_files:
            raise FileNotFoundError(f"No CellPLM h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(emb_files[0])
        embedding = adata_emb.obsm.get('X_cellplm', adata_emb.obsm.get('X_emb', adata_emb.X))
        
        logger.info(f"Loaded CellPLM embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_geneformer(self, model_result_dir):
        """Load GeneFormer embedding from CSV files."""
        emb_files = glob.glob(os.path.join(model_result_dir, '*emb.csv'))
        
        if not emb_files:
            raise FileNotFoundError(f"No GeneFormer CSV files found in {model_result_dir}")
        
        logger.info(f"Found {len(emb_files)} GeneFormer embedding files")
        
        dfs = [pd.read_csv(f, index_col=0) for f in sorted(emb_files)]
        df = pd.concat(dfs, axis=0)
        
        if 'cell_id' in df.columns:
            df['cell_id'] = df['cell_id'].astype(str)
            self.adata.obs['cell_id'] = self.adata.obs['cell_id'].astype(str)
            df_aligned = df.set_index('cell_id').loc[self.adata.obs['cell_id']]
            embedding = df_aligned.iloc[:, :1152].to_numpy()
        else:
            embedding = df.iloc[:, :1152].to_numpy()
        
        logger.info(f"Loaded GeneFormer embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_genecompass(self, model_result_dir):
        """Load GeneCompass embedding from NPY file."""
        npy_files = glob.glob(os.path.join(model_result_dir, 'cell_embeddings.npy'))
        
        if not npy_files:
            npy_files = glob.glob(os.path.join(model_result_dir, '*embeddings.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No GeneCompass NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded GeneCompass embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_scfoundation(self, model_result_dir):
        """Load scFoundation embedding from NPY file."""
        npy_files = glob.glob(os.path.join(model_result_dir, 'benchmark_*.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No scFoundation NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded scFoundation embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_sccello(self, model_result_dir):
        """Load scCello embedding from NPY file."""
        emb_dir = os.path.join(model_result_dir, 'embeddings')
        npy_files = glob.glob(os.path.join(emb_dir, '*.npy'))
        
        if not npy_files:
            npy_files = glob.glob(os.path.join(model_result_dir, '*.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No scCello NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded scCello embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_nicheformer(self, model_result_dir):
        """Load NicheFormer embedding from h5ad."""
        h5ad_files = glob.glob(os.path.join(model_result_dir, '*nicheformer*.h5ad'))
        
        if not h5ad_files:
            h5ad_files = glob.glob(os.path.join(model_result_dir, '*.h5ad'))
        
        if not h5ad_files:
            raise FileNotFoundError(f"No NicheFormer h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(h5ad_files[0])
        embedding = adata_emb.obsm.get('X_nicheformer', adata_emb.obsm.get('X_emb', adata_emb.X))
        
        logger.info(f"Loaded NicheFormer embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_scgpt(self, model_result_dir):
        """Load scGPT embedding."""
        try:
            import scgpt as scg
            
            logger.info("Running scGPT embedding extraction...")
            
            # Get GPU from config or use default
            gpu_id = self.config.get('model_paths', {}).get('scgpt', {}).get('gpu_id', 7)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Get model dir from config or use default
            model_dir = self.config.get('model_paths', {}).get('scgpt', {}).get('model_dir')
            if not model_dir:
                raise ValueError("scGPT model_dir not found in config")
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"scGPT model directory not found: {model_dir}")
            
            gene_col = "original_gene_symbols"
            if gene_col not in self.adata.var.columns:
                gene_col = self.adata.var.index.name or "index"
            
            adata_emb = scg.tasks.embed_data(
                self.adata,
                model_dir,
                gene_col=gene_col,
                batch_size=128,
            )
            
            embedding = adata_emb.obsm.get('X_scGPT', adata_emb.obsm.get('X_emb', adata_emb.X)).copy()
            self.adata = adata_emb
            
            logger.info(f"Loaded scGPT embedding: {embedding.shape}")
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to load scGPT: {e}")
            raise
    
    # ===========================
    # INTEGRATION METHOD LOADERS
    # ===========================
    
    def load_embedding_pca(self, model_result_dir):
        """Load PCA embedding from existing adata."""
        if 'X_pca' not in self.adata.obsm:
            raise ValueError("PCA not found in adata.obsm")
        
        embedding = self.adata.obsm['X_pca'].copy()
        logger.info(f"Loaded PCA embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_scvi(self, model_result_dir):
        """Run scVI integration."""
        try:
            import scib
            logger.info("Running scVI integration...")
            
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.scvi(self.adata, batch="assay", hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("scVI integration failed: X_emb not found")
            
            self.adata = adata_integrated
            embedding = adata_integrated.obsm['X_emb']
            logger.info(f"Loaded scVI embedding: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"scVI integration failed: {e}")
            raise
    
    def load_embedding_harmony(self, model_result_dir):
        """Run Harmony integration."""
        try:
            import scib
            logger.info("Running Harmony integration...")
            
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.harmony(self.adata, batch="assay", hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("Harmony integration failed: X_emb not found")
            
            self.adata = adata_integrated
            embedding = adata_integrated.obsm['X_emb']
            logger.info(f"Loaded Harmony embedding: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Harmony integration failed: {e}")
            raise
    
    def load_embedding_scanorama(self, model_result_dir):
        """Run Scanorama integration."""
        try:
            import scib
            logger.info("Running Scanorama integration...")
            
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.scanorama(self.adata, batch="assay", hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("Scanorama integration failed: X_emb not found")
            
            self.adata = adata_integrated
            embedding = adata_integrated.obsm['X_emb']
            logger.info(f"Loaded Scanorama embedding: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Scanorama integration failed: {e}")
            raise
    
    def load_embedding(self, model_name):
        """Load embedding based on model name."""
        model_name_lower = model_name.lower()
        
        if model_name_lower in ['scvi', 'harmony', 'scanorama']:
            if model_name_lower == 'scvi':
                embedding = self.load_embedding_scvi(None)
            elif model_name_lower == 'harmony':
                embedding = self.load_embedding_harmony(None)
            elif model_name_lower == 'scanorama':
                embedding = self.load_embedding_scanorama(None)
        else:
            model_result_dir = os.path.join(self.result_dir, model_name)
            
            if not os.path.exists(model_result_dir):
                raise FileNotFoundError(f"Result directory not found: {model_result_dir}")
            
            logger.info(f"Loading {model_name} embedding from {model_result_dir}")
            
            if model_name_lower == 'uce':
                embedding = self.load_embedding_uce(model_result_dir)
            elif model_name_lower == 'cellplm':
                embedding = self.load_embedding_cellplm(model_result_dir)
            elif model_name_lower == 'geneformer':
                embedding = self.load_embedding_geneformer(model_result_dir)
            elif model_name_lower == 'genecompass':
                embedding = self.load_embedding_genecompass(model_result_dir)
            elif model_name_lower == 'scfoundation':
                embedding = self.load_embedding_scfoundation(model_result_dir)
            elif model_name_lower == 'sccello':
                embedding = self.load_embedding_sccello(model_result_dir)
            elif model_name_lower == 'nicheformer':
                embedding = self.load_embedding_nicheformer(model_result_dir)
            elif model_name_lower == 'scgpt':
                embedding = self.load_embedding_scgpt(model_result_dir)
            elif model_name_lower == 'pca':
                embedding = self.load_embedding_pca(model_result_dir)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        if embedding.shape[0] != self.adata.n_obs:
            raise ValueError(f"Embedding shape mismatch: {embedding.shape[0]} vs {self.adata.n_obs}")
        
        return embedding
    
    def integrate_embedding(self, embedding):
        """Integrate embedding into adata.obsm."""
        self.adata.obsm['X_emb'] = embedding
        logger.info("Integrated embedding into adata.obsm['X_emb']")
    
    def compute_neighbors_umap(self):
        """Compute neighbors and UMAP from embedding."""
        logger.info("Computing neighbors and UMAP...")
        sc.pp.neighbors(self.adata, use_rep='X_emb', n_neighbors=15)
        sc.tl.umap(self.adata)
    
    def visualize(self, model_name):
        """Create UMAP visualizations."""
        logger.info("Creating UMAP visualizations...")
        
        viz_columns = []
        if 'batch_key' in self.adata.obs.columns:
            viz_columns.append('batch_key')
        if 'cell_type' in self.adata.obs.columns:
            viz_columns.append('cell_type')
        
        if not viz_columns:
            logger.warning("No visualization columns found")
            return
        
        model_result_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        for col in viz_columns:
            try:
                sc.pl.umap(self.adata, color=col, show=False, size=10)
                output_file = os.path.join(model_result_dir, f'umap_{col}.png')
                plt.savefig(output_file, bbox_inches='tight', dpi=150)
                plt.close()
                logger.info(f"Saved: umap_{col}.png")
            except Exception as e:
                logger.warning(f"Failed to visualize {col}: {e}")
    
    def save_results(self, model_name):
        """Save integrated data."""
        logger.info("Saving results...")
        
        model_result_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        output_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        self.adata.write_h5ad(output_file, compression='gzip')
        logger.info(f"Saved: {output_file}")
    
    def integrate(self, model_name):
        """Full integration pipeline."""
        logger.info(f"{'='*60}")
        logger.info(f"Integrating: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        self.load_data()
        embedding = self.load_embedding(model_name)
        self.integrate_embedding(embedding)
        self.compute_neighbors_umap()
        self.visualize(model_name)
        self.save_results(model_name)
        
        logger.info(f"Integration complete for {model_name.upper()}\n")


def main():
    parser = argparse.ArgumentParser(description='Integrate embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                        choices=['uce', 'cellplm', 'geneformer', 'genecompass', 'scfoundation', 
                                'sccello', 'nicheformer', 'scgpt', 'pca', 'scvi', 'harmony', 'scanorama'],
                        help='Model or integration method')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Load configuration
    config = load_config(args.config)
    
    # Get dataset-specific paths from config
    dataset_config = config['datasets'][args.dataset]
    dataset_dir = dataset_config['output_data_dir']
    result_dir = dataset_config.get('output_res_dir', 
                                    os.path.join(dataset_dir, 'Result'))
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data dir: {dataset_dir}")
    logger.info(f"Result dir: {result_dir}")
    
    try:
        integrator = EmbeddingIntegrator(dataset_dir, result_dir, config)
        integrator.integrate(args.model)
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    main()