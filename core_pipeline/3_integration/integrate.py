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
    
    def __init__(self, dataset_dir, result_dir, config, dataset_name):
        """
        Initialize integrator.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad
            result_dir: Result directory containing embeddings
            config: Configuration dictionary
            dataset_name: Name of the dataset
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_config = config['datasets'][dataset_name]
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
        emb_file = os.path.join(model_result_dir, 'adata_uce_adata.h5ad')
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"No UCE embedding h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(emb_file)
        embedding = adata_emb.obsm['X_emb']
        self.adata.obsm['X_emb'] = embedding

        logger.info(f"Integrated UCE embedding: {embedding.shape}")
    

    def load_embedding_cellplm(self, model_result_dir):
        """Load CellPLM embedding by running model inference."""
        try:
            module_path = self.config.get('model_paths', {}).get('cellplm', {}).get('module_path')
            if not module_path:
                raise ValueError("CellPLM module_path not found in config")
            
            sys.path.insert(0, module_path)
            from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
            
            logger.info("Running CellPLM embedding inference...")
            
            gpu_id = self.config.get('model_paths', {}).get('cellplm', {}).get('gpu_id', 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            pretrain_version = self.config.get('model_paths', {}).get('cellplm', {}).get('pretrain_version', '20231027_85M')
            pretrain_directory = self.config.get('model_paths', {}).get('cellplm', {}).get('pretrain_directory')
            batch_size = self.config.get('model_paths', {}).get('cellplm', {}).get('batch_size', 2048)
            
            if not pretrain_directory:
                raise ValueError("CellPLM pretrain_directory not found in config")
            
            device = f'cuda:{gpu_id}'
            
            pipeline = CellEmbeddingPipeline(
                pretrain_prefix=pretrain_version,
                pretrain_directory=pretrain_directory
            )
            
            embedding = pipeline.predict(
                self.adata,
                device=device,
                inference_config={"batch_size": batch_size},
                ensembl_auto_conversion=False
            )
            
            embedding = embedding.cpu().numpy()
            self.adata.obsm['X_emb'] = embedding
            logger.info(f"Integrated CellPLM embedding: {embedding.shape}")
        
        except Exception as e:
            logger.error(f"Failed to load CellPLM embedding: {e}")
            raise


    def load_embedding_geneformer(self, model_result_dir):
        """Load GeneFormer embedding from CSV files."""
        emb_files = glob.glob(os.path.join(model_result_dir, '*emb.csv'))
        
        if not emb_files:
            raise FileNotFoundError(f"No GeneFormer CSV files found in {model_result_dir}")
        
        logger.info(f"Found {len(emb_files)} GeneFormer embedding files")
        
        embedding_dim = self.config.get('model_paths', {}).get('geneformer', {}).get('embedding_dim', 1152)
        
        dfs = [pd.read_csv(f, index_col=0) for f in sorted(emb_files)]
        df = pd.concat(dfs, axis=0)
        
        if 'cell_id' in df.columns:
            df['cell_id'] = df['cell_id'].astype(str)
            self.adata.obs['cell_id'] = self.adata.obs['cell_id'].astype(str)
            df_aligned = df.set_index('cell_id').loc[self.adata.obs['cell_id']]
            embedding = df_aligned.iloc[:, :embedding_dim].to_numpy()
        else:
            embedding = df.iloc[:, :embedding_dim].to_numpy()
        
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated GeneFormer embedding: {embedding.shape}")
    
    
    def load_embedding_genecompass(self, model_result_dir):
        """Load GeneCompass embedding from NPY file."""
        npy_file = os.path.join(model_result_dir, 'cell_embeddings.npy')
        
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"No GeneCompass NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_file)
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated GeneCompass embedding: {embedding.shape}")
    
    def load_embedding_scfoundation(self, model_result_dir):
        """Load scFoundation embedding from NPY file."""
        npy_file = os.path.join(model_result_dir, 'benchmark_*.npy')
        
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"No scFoundation NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_file)
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated scFoundation embedding: {embedding.shape}")
    
    def load_embedding_sccello(self, model_result_dir):
        """Load scCello embedding from NPY file."""
        emb_dir = os.path.join(model_result_dir, 'embeddings')
        npy_file = os.path.join(emb_dir, '*.npy')
        
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"No scCello NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_file)
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated scCello embedding: {embedding.shape}")
    
    def load_embedding_nicheformer(self, model_result_dir):
        """Load NicheFormer embedding from h5ad."""
        h5ad_files = glob.glob(os.path.join(model_result_dir, '*nicheformer*.h5ad'))
        
        if not h5ad_files:
            raise FileNotFoundError(f"No NicheFormer h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(h5ad_files[0])
        embedding = adata_emb.obsm['X_emb']

        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated NicheFormer embedding: {embedding.shape}")
    

    def load_embedding_scgpt(self, model_result_dir):
        """Load scGPT embedding."""
        try:
            import scgpt as scg
            
            logger.info("Running scGPT embedding extraction...")
            
            gpu_id = self.config.get('model_paths', {}).get('scgpt', {}).get('gpu_id', 6)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            model_dir = self.config.get('model_paths', {}).get('scgpt', {}).get('model_dir')
            batch_size = self.config.get('model_paths', {}).get('scgpt', {}).get('batch_size', 32)

            if not model_dir:
                raise ValueError("scGPT model_dir not found in config")
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"scGPT model directory not found: {model_dir}")
            
            gene_col = self.dataset_config.get('gene_col', 'original_gene_symbols')
            if gene_col not in self.adata.var.columns:
                gene_col = self.adata.var.index.name or "index"
                logger.warning(f"gene_col '{gene_col}' not found, using: {gene_col}")
            
            adata_emb = scg.tasks.embed_data(
                self.adata,
                model_dir,
                gene_col=gene_col,
                batch_size=batch_size,
            )
            
            embedding = adata_emb.obsm.get('X_scGPT', adata_emb.obsm.get('X_emb', adata_emb.X)).copy()
            self.adata = adata_emb
            self.adata.obsm['X_emb'] = embedding
            
            logger.info(f"Integrated scGPT embedding: {embedding.shape}")
        
        except Exception as e:
            logger.error(f"Failed to load scGPT: {e}")
            raise
    
    def load_embedding_pca(self, model_result_dir):
        """Load PCA embedding from existing adata."""
        if 'X_pca' not in self.adata.obsm:
            raise ValueError("PCA not found in adata.obsm")
        
        embedding = self.adata.obsm['X_pca'].copy()
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated PCA embedding: {embedding.shape}")
    
    def load_embedding_scvi(self, model_result_dir):
        """Run scVI integration."""
        try:
            import scib
            logger.info("Running scVI integration...")
            
            batch_key = self.dataset_config.get('batch_key', 'batch')
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.scvi(self.adata, batch=batch_key, hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("scVI integration failed: X_emb not found")
            
            self.adata = adata_integrated
            logger.info(f"Integrated scVI embedding: {self.adata.obsm['X_emb'].shape}")
        except Exception as e:
            logger.error(f"scVI integration failed: {e}")
            raise
    
    def load_embedding_harmony(self, model_result_dir):
        """Run Harmony integration."""
        try:
            import scib
            logger.info("Running Harmony integration...")
            
            batch_key = self.dataset_config.get('batch_key', 'batch')
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.harmony(self.adata, batch=batch_key, hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("Harmony integration failed: X_emb not found")
            
            self.adata = adata_integrated
            logger.info(f"Integrated Harmony embedding: {self.adata.obsm['X_emb'].shape}")
        except Exception as e:
            logger.error(f"Harmony integration failed: {e}")
            raise
    
    def load_embedding_scanorama(self, model_result_dir):
        """Run Scanorama integration."""
        try:
            import scib
            logger.info("Running Scanorama integration...")
            
            batch_key = self.dataset_config.get('batch_key', 'batch')
            hvg = self.adata.var[self.adata.var.get('highly_variable', False)].index.tolist() if 'highly_variable' in self.adata.var else None
            
            adata_integrated = scib.integration.scanorama(self.adata, batch=batch_key, hvg=hvg)
            
            if 'X_emb' not in adata_integrated.obsm:
                raise ValueError("Scanorama integration failed: X_emb not found")
            
            self.adata = adata_integrated
            logger.info(f"Integrated Scanorama embedding: {self.adata.obsm['X_emb'].shape}")
        except Exception as e:
            logger.error(f"Scanorama integration failed: {e}")
            raise
    
    def load_and_integrate_embedding(self, model_name):
        """Load and integrate embedding in one step."""
        model_name_lower = model_name.lower()
        
        # For integration methods, model_result_dir is not used but parameter is required
        model_result_dir = os.path.join(self.result_dir, model_name)
        
        if model_name_lower == 'uce':
            self.load_embedding_uce(model_result_dir)
        elif model_name_lower == 'cellplm':
            self.load_embedding_cellplm(model_result_dir)
        elif model_name_lower == 'geneformer':
            self.load_embedding_geneformer(model_result_dir)
        elif model_name_lower == 'genecompass':
            self.load_embedding_genecompass(model_result_dir)
        elif model_name_lower == 'scfoundation':
            self.load_embedding_scfoundation(model_result_dir)
        elif model_name_lower == 'sccello':
            self.load_embedding_sccello(model_result_dir)
        elif model_name_lower == 'nicheformer':
            self.load_embedding_nicheformer(model_result_dir)
        elif model_name_lower == 'scgpt':
            self.load_embedding_scgpt(model_result_dir)
        elif model_name_lower == 'pca':
            self.load_embedding_pca(model_result_dir)
        elif model_name_lower == 'scvi':
            self.load_embedding_scvi(model_result_dir)
        elif model_name_lower == 'harmony':
            self.load_embedding_harmony(model_result_dir)
        elif model_name_lower == 'scanorama':
            self.load_embedding_scanorama(model_result_dir)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if 'X_emb' not in self.adata.obsm:
            raise ValueError("Embedding not found in adata.obsm['X_emb']")
        
        if self.adata.obsm['X_emb'].shape[0] != self.adata.n_obs:
            raise ValueError(f"Embedding shape mismatch: {self.adata.obsm['X_emb'].shape[0]} vs {self.adata.n_obs}")
        
        
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
        self.load_and_integrate_embedding(model_name)
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
        integrator = EmbeddingIntegrator(dataset_dir, result_dir, config, args.dataset)
        integrator.integrate(args.model)
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise