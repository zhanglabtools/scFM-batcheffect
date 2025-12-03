#!/usr/bin/env python3
"""
Extract NicheFormer embeddings

NicheFormer is a single-cell foundation model trained on spatial transcriptomics data
It uses hierarchical attention and preserves spatial neighborhood information

Input: {output_data_dir}/geneformer/adata.h5ad
Output: {output_res_dir}/nicheformer/adata_nicheformer.h5ad
"""

import argparse
import os
import sys
import yaml
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import anndata as ad
from tqdm import tqdm

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


def align_to_model(model, adata, fill_value=0.0):
    """
    Align adata to model's feature (gene) set, ensuring column order matches model.
    Missing genes are padded with zeros.
    
    Args:
        model: Reference AnnData object with target feature order
        adata: AnnData object to align
        fill_value: Fill value for missing genes (default: 0)
    
    Returns:
        Aligned AnnData object
    """
    model_genes = model.var_names
    adata_genes = adata.var_names
    
    # Find missing genes
    missing_genes = [g for g in model_genes if g not in adata_genes]
    
    # Add missing genes as zero columns
    if missing_genes:
        n_missing = len(missing_genes)
        zero_mat = np.full((adata.n_obs, n_missing), fill_value)
        
        zero_adata = ad.AnnData(
            X=zero_mat,
            var=pd.DataFrame(index=missing_genes),
            obs=adata.obs.copy()
        )
        
        # Concatenate
        adata = ad.concat([adata, zero_adata], axis=1, join='outer')
    
    # Reorder columns to match model
    adata_aligned = adata[:, model_genes].copy()
    
    return adata_aligned


def extract_nicheformer(dataset_name, config_path):
    """
    Extract NicheFormer embeddings.
    
    Args:
        dataset_name: Dataset name (e.g., 'limb', 'liver')
        config_path: Path to config.yaml
    """
    logger.info("Starting NicheFormer embedding extraction")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get dataset config
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config")
    
    dataset_config = config['datasets'][dataset_name]
    output_data_dir = dataset_config['output_data_dir']
    output_res_dir = dataset_config.get('output_res_dir',
                                        os.path.join(output_data_dir, 'Result', dataset_name))
    
    # Get model config
    nf_config = config['model_paths']['nicheformer']
    model_checkpoint = nf_config['model_dir']
    technology_mean_path = nf_config['technology_mean_path']
    data_model_path = nf_config['model_reference_path']
    gpu_id = nf_config.get('gpu', 0)
    batch_size = nf_config.get('batch_size', 32)
    max_seq_len = nf_config.get('max_seq_len', 1500)
    aux_tokens = nf_config.get('aux_tokens', 30)
    chunk_size = nf_config.get('chunk_size', 1000)
    num_workers = nf_config.get('num_workers', 4)
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data dir: {output_data_dir}")
    logger.info(f"Result dir: {output_res_dir}")
    logger.info(f"GPU: {gpu_id}, Batch size: {batch_size}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Check input file
    input_file = os.path.join(output_data_dir, 'geneformer', 'adata.h5ad')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"GeneFormer h5ad not found at {input_file}")
    
    logger.info(f"Loading input data from {input_file}...")
    adata = ad.read_h5ad(input_file)
    logger.info(f"Input data shape: {adata.shape}")
    
    # Create output directory
    result_dir = os.path.join(output_res_dir, 'nicheformer')
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Output directory: {result_dir}")
    
    # Check model resources exist
    for path, name in [(model_checkpoint, 'Model checkpoint'),
                       (technology_mean_path, 'Technology mean'),
                       (data_model_path, 'Model reference')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}")
    
    logger.info("Loading model reference and technology mean...")
    adata_model = ad.read_h5ad(data_model_path)
    technology_mean = np.load(technology_mean_path)
    
    # Align input data to model's gene set
    logger.info("Aligning data to model gene set...")
    adata_aligned = align_to_model(adata_model, adata)
    logger.info(f"Aligned data shape: {adata_aligned.shape}")
    
    # Add required metadata columns
    adata_aligned.obs['modality'] = nf_config.get('modality_code', 3)
    adata_aligned.obs['specie'] = nf_config.get('specie_code', 5)
    adata_aligned.obs['assay'] = nf_config.get('assay_code', 12)
    adata_aligned.obs['nicheformer_split'] = 'train'
    
    logger.info("Importing NicheFormer...")
    code_path = nf_config.get('code_path')
    if code_path and code_path not in sys.path:
        sys.path.insert(0, code_path)
    
    try:
        from nicheformer.models import Nicheformer
        from nicheformer.data import NicheformerDataset
    except ImportError as e:
        raise ImportError(f"Failed to import NicheFormer: {e}")
    
    logger.info("Creating NicheFormer dataset...")
    
    # Create dataset
    dataset = NicheformerDataset(
        adata=adata_aligned,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=max_seq_len,
        aux_tokens=aux_tokens,
        chunk_size=chunk_size,
        metadata_fields={'obs': ['modality', 'specie', 'assay']}
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info("Loading NicheFormer checkpoint...")
    
    # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        strict=False
    )
    model.eval()
    
    logger.info("Extracting embeddings...")
    embeddings = []
    device = model.embeddings.weight.device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting NicheFormer embeddings"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Get embeddings from the model
            emb = model.get_embeddings(
                batch=batch,
                layer=-1
            )
            embeddings.append(emb.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Store in adata
    adata.obsm["X_nicheformer"] = embeddings
    
    # Save result
    output_file = os.path.join(result_dir, "adata_nicheformer.h5ad")
    adata.write_h5ad(output_file, compression='gzip')
    logger.info(f"Saved output to {output_file}")
    
    logger.info("✓ NicheFormer embedding extraction complete")
    logger.info(f"Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract NicheFormer embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    try:
        extract_nicheformer(args.dataset, args.config)
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    main()
