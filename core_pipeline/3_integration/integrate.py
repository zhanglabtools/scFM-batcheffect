#!/usr/bin/env python3
"""
Integration script: Combine embeddings with original AnnData object

For each model, loads its embedding from Result directory and:
1. Loads original standardized data (data.h5ad)
2. Loads embedding from Result/{model}/ with model-specific logic
3. Integrates embedding into adata.obsm['X_emb']
4. Performs UMAP visualization on embedding space
5. Saves integrated data with visualizations

Supports 12 models/algorithms:
- Embedding models: UCE, CellPLM, GeneFormer, GeneCompass, scFoundation, scCello, NicheFormer
- Integration methods: scVI, Harmony, Scanorama, PCA

Usage:
    python integrate.py --dataset {dataset_dir} --model {model_name}
    
Examples:
    python integrate.py --dataset /path/to/limb --model geneformer
    python integrate.py --dataset /path/to/liver --model genecompass --result-dir /custom/path
    python integrate.py --dataset /path/to/immune --model harmony
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
import glob
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Models that require preprocessing
MODELS_REQUIRING_PREPROCESSING = {
    'uce': 'requires_normalization',  # Need normalize + log1p
    'cellplm': 'requires_normalization',
    'scgpt': 'no_preprocessing',  # Already processed
}

MODELS_REQUIRING_HVG = {
    'scvi', 'harmony', 'scanorama'  # Integration methods that need HVG
}


class EmbeddingIntegrator:
    """Integrate embeddings from different models."""
    
    def __init__(self, dataset_dir, result_dir):
        """
        Initialize integrator.
        
        Args:
            dataset_dir: Directory containing standardized data.h5ad
            result_dir: Result directory containing embeddings
        """
        self.dataset_dir = dataset_dir
        self.result_dir = result_dir
        self.adata = None
    
    def load_data(self):
        """Load original standardized data."""
        data_file = os.path.join(self.dataset_dir, 'data.h5ad')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"data.h5ad not found at {data_file}")
        
        logger.info(f"Loading original data from {data_file}")
        self.adata = sc.read_h5ad(data_file)
        logger.info(f"Data shape: {self.adata.shape}")
        
        return self.adata
    
    # ===========================
    # EMBEDDING MODEL LOADERS
    # ===========================
    
    def load_embedding_uce(self, model_result_dir):
        """Load UCE (Universal Cell Embeddings) from h5ad."""
        emb_files = glob.glob(os.path.join(model_result_dir, 'adata_uce_*.h5ad'))
        
        if not emb_files:
            raise FileNotFoundError(f"No UCE embedding h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(emb_files[0])
        
        if 'X_uce' in adata_emb.obsm:
            embedding = adata_emb.obsm['X_uce']
        else:
            embedding = adata_emb.X
        
        logger.info(f"Loaded UCE embedding: {embedding.shape} (1280-dim)")
        return embedding
    
    def load_embedding_cellplm(self, model_result_dir):
        """Load CellPLM embedding from preprocessed h5ad or via pipeline."""
        logger.info("Loading CellPLM embedding...")
        
        try:
            # Try to find preprocessed embedding h5ad first
            emb_files = glob.glob(os.path.join(model_result_dir, '*.h5ad'))
            
            if emb_files:
                # Load from existing h5ad file
                adata_emb = sc.read_h5ad(emb_files[0])
                if 'X_cellplm' in adata_emb.obsm:
                    embedding = adata_emb.obsm['X_cellplm']
                elif 'X_emb' in adata_emb.obsm:
                    embedding = adata_emb.obsm['X_emb']
                else:
                    embedding = adata_emb.X
                logger.info(f"Loaded CellPLM embedding from h5ad: {embedding.shape} (768-dim)")
                return embedding
            
            # Otherwise, try dynamic loading via pipeline
            # Import config loader at top if not already done
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            try:
                from config_loader import load_config, get_model_path, get_model_param
                config = load_config()
            except:
                config = None
                logger.warning("Could not load config.yaml, using hardcoded defaults for CellPLM")
            
            # Get paths from config or use defaults
            if config:
                cellplm_code_path = get_model_path(config, 'cellplm', 'code_path',
                                                   '/home/wanglinting/LCBERT/Code/model-cellplm')
                cellplm_checkpoint_dir = get_model_path(config, 'cellplm', 'checkpoint_dir',
                                                        '/home/wanglinting/LCBERT/Code/model-cellplm/ckpt')
                gpu_id = get_model_param(config, 'cellplm', 'gpu_id', default=4)
                pretrain_version = get_model_param(config, 'cellplm', 'pretrain_version', default='20231027_85M')
            else:
                cellplm_code_path = '/home/wanglinting/LCBERT/Code/model-cellplm'
                cellplm_checkpoint_dir = '/home/wanglinting/LCBERT/Code/model-cellplm/ckpt'
                gpu_id = 4
                pretrain_version = '20231027_85M'
            
            sys.path.insert(0, cellplm_code_path)
            from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
            
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            pipeline = CellEmbeddingPipeline(
                pretrain_prefix=pretrain_version,
                pretrain_directory=cellplm_checkpoint_dir
            )
            
            embedding = pipeline.predict(
                self.adata,
                device=f'cuda:0',
                inference_config={"batch_size": 2048},
                ensembl_auto_conversion=False
            )
            
            logger.info(f"Loaded CellPLM embedding: {embedding.shape} (768-dim)")
            return embedding.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Failed to load CellPLM: {e}")
            raise
    
    def load_embedding_geneformer(self, model_result_dir):
        """Load GeneFormer embedding from CSV files (1152-dim)."""
        emb_files = glob.glob(os.path.join(model_result_dir, '*emb.csv'))
        
        if not emb_files:
            raise FileNotFoundError(f"No GeneFormer CSV files found in {model_result_dir}")
        
        logger.info(f"Found {len(emb_files)} GeneFormer embedding files")
        
        # Load and concatenate CSV files
        dfs = [pd.read_csv(f, index_col=0) for f in sorted(emb_files)]
        df = pd.concat(dfs, axis=0)
        
        # Align by cell_id
        if 'cell_id' in df.columns:
            df['cell_id'] = df['cell_id'].astype(str)
            self.adata.obs['cell_id'] = self.adata.obs['cell_id'].astype(str)
            df_aligned = df.set_index('cell_id').loc[self.adata.obs['cell_id']]
            # Extract first 1152 dimensions (hidden_size)
            embedding = df_aligned.iloc[:, :1152].to_numpy()
        else:
            # If no cell_id column, assume CSV is already aligned
            embedding = df.iloc[:, :1152].to_numpy()
        
        logger.info(f"Loaded GeneFormer embedding: {embedding.shape} (1152-dim)")
        return embedding
    
    def load_embedding_genecompass(self, model_result_dir):
        """Load GeneCompass embedding from NPY file (768-dim)."""
        # Try multiple patterns
        npy_files = glob.glob(os.path.join(model_result_dir, 'cell_embeddings.npy'))
        
        if not npy_files:
            npy_files = glob.glob(os.path.join(model_result_dir, '*embeddings.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No GeneCompass NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded GeneCompass embedding: {embedding.shape} (768-dim)")
        return embedding
    
    def load_embedding_scfoundation(self, model_result_dir):
        """Load scFoundation embedding from NPY file (512-dim)."""
        # scFoundation names output as benchmark_*.npy
        npy_files = glob.glob(os.path.join(model_result_dir, 'benchmark_*.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No scFoundation NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded scFoundation embedding: {embedding.shape} (512-dim)")
        return embedding
    
    def load_embedding_sccello(self, model_result_dir):
        """Load scCello embedding from NPY file (768-dim)."""
        # scCello saves in embeddings/ subdirectory
        emb_dir = os.path.join(model_result_dir, 'embeddings')
        npy_files = glob.glob(os.path.join(emb_dir, '*.npy'))
        
        if not npy_files:
            # Try direct npy in result dir
            npy_files = glob.glob(os.path.join(model_result_dir, '*.npy'))
        
        if not npy_files:
            raise FileNotFoundError(f"No scCello NPY file found in {model_result_dir}")
        
        embedding = np.load(npy_files[0])
        logger.info(f"Loaded scCello embedding: {embedding.shape} (768-dim)")
        return embedding
    
    def load_embedding_nicheformer(self, model_result_dir):
        """Load NicheFormer embedding from h5ad (768-dim)."""
        h5ad_files = glob.glob(os.path.join(model_result_dir, '*nicheformer*.h5ad'))
        
        if not h5ad_files:
            h5ad_files = glob.glob(os.path.join(model_result_dir, '*.h5ad'))
        
        if not h5ad_files:
            raise FileNotFoundError(f"No NicheFormer h5ad found in {model_result_dir}")
        
        adata_emb = sc.read_h5ad(h5ad_files[0])
        
        if 'X_nicheformer' in adata_emb.obsm:
            embedding = adata_emb.obsm['X_nicheformer']
        elif 'X_emb' in adata_emb.obsm:
            embedding = adata_emb.obsm['X_emb']
        else:
            embedding = adata_emb.X
        
        logger.info(f"Loaded NicheFormer embedding: {embedding.shape} (768-dim)")
        return embedding
    
    def load_embedding_scgpt(self, model_result_dir):
        """Load scGPT embedding via scGPT.tasks.embed_data()."""
        try:
            import scgpt as scg
            
            logger.info("Running scGPT embedding extraction...")
            
            # Set GPU
            GPU_NUMBER = [7]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
            
            # scGPT model directory
            model_dir = "/home/wanglinting/LCBERT/Download/scGPT/model/scGPT_human"
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"scGPT model directory not found: {model_dir}")
            
            # Gene column name (adjust based on your data)
            gene_col = "original_gene_symbols"
            if gene_col not in self.adata.var.columns:
                gene_col = self.adata.var.index.name or "index"
            
            # Extract embedding
            adata_emb = scg.tasks.embed_data(
                self.adata,
                model_dir,
                gene_col=gene_col,
                batch_size=128,
            )
            
            # Get X_scGPT embedding
            if 'X_scGPT' in adata_emb.obsm:
                embedding = adata_emb.obsm['X_scGPT'].copy()
            elif 'X_emb' in adata_emb.obsm:
                embedding = adata_emb.obsm['X_emb'].copy()
            else:
                embedding = adata_emb.X.copy()
            
            # Update self.adata with scGPT-processed data
            self.adata = adata_emb
            
            logger.info(f"Loaded scGPT embedding: {embedding.shape} (768-dim)")
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
            raise ValueError("PCA not found in adata.obsm. Ensure data was preprocessed with PCA.")
        
        embedding = self.adata.obsm['X_pca'].copy()
        logger.info(f"Loaded PCA embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_scvi(self, model_result_dir):
        """Run scVI integration on the fly."""
        import scib
        
        logger.info("Running scVI integration...")
        
        # Get HVG if available
        if 'highly_variable' in self.adata.var:
            hvg = self.adata.var[self.adata.var['highly_variable']].index.tolist()
        else:
            hvg = None
        
        # Run scVI
        adata_integrated = scib.integration.scvi(
            self.adata,
            batch="assay",
            hvg=hvg
        )
        
        if 'X_emb' not in adata_integrated.obsm:
            raise ValueError("scVI integration failed: X_emb not found")
        
        # Update self.adata with integrated result
        self.adata = adata_integrated
        
        embedding = adata_integrated.obsm['X_emb']
        logger.info(f"Loaded scVI embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_harmony(self, model_result_dir):
        """Run Harmony integration on the fly."""
        import scib
        
        logger.info("Running Harmony integration...")
        
        # Set thread number
        default_n_threads = 32
        os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
        os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
        os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
        
        # Get HVG if available
        if 'highly_variable' in self.adata.var:
            hvg = self.adata.var[self.adata.var['highly_variable']].index.tolist()
        else:
            hvg = None
        
        # Run Harmony
        adata_integrated = scib.integration.harmony(
            self.adata,
            batch="assay",
            hvg=hvg
        )
        
        if 'X_emb' not in adata_integrated.obsm:
            raise ValueError("Harmony integration failed: X_emb not found")
        
        # Update self.adata with integrated result
        self.adata = adata_integrated
        
        embedding = adata_integrated.obsm['X_emb']
        logger.info(f"Loaded Harmony embedding: {embedding.shape}")
        return embedding
    
    def load_embedding_scanorama(self, model_result_dir):
        """Run Scanorama integration on the fly."""
        import scib
        
        logger.info("Running Scanorama integration...")
        
        # Set thread number
        default_n_threads = 32
        os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
        os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
        os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
        
        # Get HVG if available
        if 'highly_variable' in self.adata.var:
            hvg = self.adata.var[self.adata.var['highly_variable']].index.tolist()
        else:
            hvg = None
        
        # Run Scanorama
        adata_integrated = scib.integration.scanorama(
            self.adata,
            batch="assay",
            hvg=hvg
        )
        
        if 'X_emb' not in adata_integrated.obsm:
            raise ValueError("Scanorama integration failed: X_emb not found")
        
        # Update self.adata with integrated result
        self.adata = adata_integrated
        
        embedding = adata_integrated.obsm['X_emb']
        logger.info(f"Loaded Scanorama embedding: {embedding.shape}")
        return embedding
    
    def load_embedding(self, model_name):
        """Load embedding based on model name."""
        model_name_lower = model_name.lower()
        
        # For integration methods that don't use pre-computed embeddings
        if model_name_lower in ['scvi', 'harmony', 'scanorama']:
            # These methods compute embeddings on the fly
            if model_name_lower == 'scvi':
                embedding = self.load_embedding_scvi(None)
            elif model_name_lower == 'harmony':
                embedding = self.load_embedding_harmony(None)
            elif model_name_lower == 'scanorama':
                embedding = self.load_embedding_scanorama(None)
        else:
            # For models that have pre-computed embeddings
            model_result_dir = os.path.join(self.result_dir, model_name)
            
            if not os.path.exists(model_result_dir):
                raise FileNotFoundError(f"Result directory not found: {model_result_dir}")
            
            logger.info(f"Loading {model_name} embedding from {model_result_dir}")
            
            # Model-specific loading logic
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
        
        # Validate shape
        if embedding.shape[0] != self.adata.n_obs:
            raise ValueError(
                f"Embedding shape mismatch: {embedding.shape[0]} cells vs "
                f"{self.adata.n_obs} cells in data"
            )
        
        return embedding
    
    def integrate_embedding(self, embedding, model_name):
        """Integrate embedding into adata.obsm."""
        self.adata.obsm['X_emb'] = embedding
        logger.info(f"Integrated embedding into adata.obsm['X_emb']")
    
    def preprocess_if_needed(self, model_name):
        """Apply preprocessing specific to model requirements."""
        model_name_lower = model_name.lower()
        
        # Some models need normalization
        if model_name_lower in ['uce', 'cellplm']:
            logger.info(f"Applying preprocessing for {model_name}...")
            if 'log1p' not in self.adata.X.__class__.__name__:
                logger.info("Normalizing and log-transforming data...")
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
    
    def compute_neighbors_umap(self):
        """Compute neighbors and UMAP from embedding."""
        logger.info("Computing neighbor graph from embedding (k=15)...")
        sc.pp.neighbors(self.adata, use_rep='X_emb', n_neighbors=15)
        
        logger.info("Computing UMAP...")
        sc.tl.umap(self.adata)
    
    def visualize(self, model_name):
        """Create UMAP visualizations for batch_key and celltype_key from config."""
        logger.info("Creating UMAP visualizations...")
        
        # Try to load config.yaml to get batch_key and celltype_key
        config_file = os.path.join(self.dataset_dir, 'config.yaml')
        viz_columns = []
        
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Get batch_key and celltype_key from config
                if isinstance(config, dict):
                    if 'batch_key' in config and config['batch_key'] in self.adata.obs.columns:
                        viz_columns.append(config['batch_key'])
                    if 'celltype_key' in config and config['celltype_key'] in self.adata.obs.columns:
                        viz_columns.append(config['celltype_key'])
                
                logger.info(f"Loaded visualization columns from config: {viz_columns}")
            except Exception as e:
                logger.warning(f"Failed to load config.yaml: {e}. Using fallback columns.")
        
        if not viz_columns:
            logger.warning("No visualization columns found in obs")
            return
        
        logger.info(f"Visualizing columns: {viz_columns}")
        
        model_result_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        for col in viz_columns:
            try:
                fig = sc.pl.umap(
                    self.adata,
                    color=col,
                    show=False,
                    size=10
                )
                
                output_file = os.path.join(model_result_dir, f'umap_{col}.png')
                plt.savefig(output_file, bbox_inches='tight', dpi=150)
                plt.close()
                logger.info(f"✓ Saved: umap_{col}.png")
            except Exception as e:
                logger.warning(f"Failed to visualize {col}: {e}")
    
    def save_results(self, model_name):
        """Save integrated data and results."""
        logger.info("Saving integrated results...")
        
        model_result_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        # Save integrated h5ad
        output_file = os.path.join(model_result_dir, f'Embeddings_{model_name}.h5ad')
        self.adata.write_h5ad(output_file, compression='gzip')
        logger.info(f"✓ Saved: Embeddings_{model_name}.h5ad ({self.adata.shape})")
    
    def integrate(self, model_name):
        """Full integration pipeline."""
        logger.info(f"{'='*60}")
        logger.info(f"Starting integration for: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Load original data
        self.load_data()
        
        # Load model embedding
        embedding = self.load_embedding(model_name)
        
        # Integrate
        self.integrate_embedding(embedding, model_name)
        
        # Preprocess if needed
        self.preprocess_if_needed(model_name)
        
        # Compute neighbors and UMAP
        self.compute_neighbors_umap()
        
        # Visualize
        self.visualize(model_name)
        
        # Save
        self.save_results(model_name)
        
        logger.info(f"✓ Integration complete for {model_name.upper()}")
        logger.info(f"{'='*60}\n")

def integrate_model(dataset_dir, model_name, result_dir=None):
    """
    Integrate embeddings for a specific model.
    
    Args:
        dataset_dir: Dataset directory containing data.h5ad
        model_name: Name of the model or algorithm
        result_dir: Result directory (default: {dataset_dir}/../Result/{dataset_name})
    
    Supported models:
        Embedding models:
            - uce: Universal Cell Embeddings (1280-dim)
            - cellplm: CellPLM (768-dim)
            - geneformer: GeneFormer (1152-dim)
            - genecompass: GeneCompass (768-dim)
            - scfoundation: scFoundation (512-dim)
            - sccello: scCello (768-dim)
            - nicheformer: NicheFormer (768-dim)
            - scgpt: scGPT (768-dim)
        
        Integration methods:
            - pca: PCA from preprocessed data
            - scvi: scVI integration (uses scib)
            - harmony: Harmony integration (uses scib)
            - scanorama: Scanorama integration (uses scib)
    """
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    # Infer result directory if not provided
    if result_dir is None:
        dataset_name = os.path.basename(dataset_dir)
        result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name)
    
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Result directory: {result_dir}")
    logger.info(f"Model: {model_name}")
    
    # Create integrator and run
    try:
        integrator = EmbeddingIntegrator(dataset_dir, result_dir)
        integrator.integrate(model_name)
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        return True
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Integrate embeddings from a specific model or integration method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Embedding models
    python integrate.py --dataset /path/to/limb --model geneformer
    python integrate.py --dataset /path/to/liver --model genecompass --result-dir /custom/path
    python integrate.py --dataset /path/to/immune --model uce
    
    # Integration methods
    python integrate.py --dataset /path/to/hlca_assay --model harmony
    python integrate.py --dataset /path/to/hlca_assay --model scvi
    python integrate.py --dataset /path/to/hlca_assay --model pca

Supported Models:
    Embedding:
        - uce (1280-dim)
        - cellplm (768-dim)
        - geneformer (1152-dim)
        - genecompass (768-dim)
        - scfoundation (512-dim)
        - sccello (768-dim)
        - nicheformer (768-dim)
        - scgpt (768-dim)
    
    Integration:
        - pca
        - scvi
        - harmony
        - scanorama
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset directory containing data.h5ad'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['uce', 'cellplm', 'geneformer', 'genecompass', 'scfoundation', 
                'sccello', 'nicheformer', 'scgpt', 'pca', 'scvi', 'harmony', 'scanorama'],
        help='Model or integration method name'
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        help='Result directory (default: inferred from dataset path)'
    )
    
    args = parser.parse_args()
    
    integrate_model(args.dataset, args.model, args.result_dir)
