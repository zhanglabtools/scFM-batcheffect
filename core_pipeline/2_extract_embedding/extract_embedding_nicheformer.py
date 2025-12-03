#!/usr/bin/env python3
"""
Extract NicheFormer embeddings

NicheFormer is a single-cell foundation model trained on spatial transcriptomics data
It uses hierarchical attention and preserves spatial neighborhood information

Input: {dataset_dir}/geneformer/adata.h5ad (uses same format as GeneFormer)
Output: {dataset_dir}/../Result/{dataset_name}/nicheformer/adata_nicheformer.h5ad
"""

import os
import sys
import argparse
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

# Set timezone
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import NicheFormer
from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset


def align_to_model(model: ad.AnnData, adata: ad.AnnData, fill_value: float = 0.0) -> ad.AnnData:
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


def extract_nicheformer(dataset_dir, gpu_id=0, batch_size=32):
    """
    Extract NicheFormer embeddings.
    
    Args:
        dataset_dir: Dataset directory containing 'geneformer/adata.h5ad'
        gpu_id: GPU device ID to use
        batch_size: Inference batch size
    """
    logger.info(f"Starting NicheFormer embedding extraction from {dataset_dir}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Check input file
    input_file = os.path.join(dataset_dir, 'geneformer', 'adata.h5ad')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"GeneFormer h5ad not found at {input_file}")
    
    logger.info(f"Loading input data from {input_file}...")
    adata = ad.read_h5ad(input_file)
    logger.info(f"Input data shape: {adata.shape}")
    
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_dir)
    
    # Create output directory
    result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name, 'nicheformer')
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Output directory: {result_dir}")
    
    # Model configuration
    model_checkpoint = "/home/wanglinting/LCBERT/Download/nicheformer/nicheformer.ckpt"
    technology_mean_path = "/home/wanglinting/LCBERT/Download/nicheformer/data/model_means/dissociated_mean_script.npy"
    data_model_path = "/home/wanglinting/LCBERT/Download/nicheformer/data/model_means/model.h5ad"
    
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"NicheFormer model not found at {model_checkpoint}")
    if not os.path.exists(technology_mean_path):
        raise FileNotFoundError(f"Technology mean file not found at {technology_mean_path}")
    if not os.path.exists(data_model_path):
        raise FileNotFoundError(f"Model reference data not found at {data_model_path}")
    
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Loading reference model for gene alignment...")
    adata_model = ad.read_h5ad(data_model_path)
    technology_mean = np.load(technology_mean_path)
    
    # Align input data to model's gene set
    logger.info("Aligning data to model gene set...")
    adata_aligned = align_to_model(adata_model, adata)
    logger.info(f"Aligned data shape: {adata_aligned.shape}")
    
    # Add required metadata columns
    adata_aligned.obs['modality'] = 3  # dissociated
    adata_aligned.obs['specie'] = 5  # human
    adata_aligned.obs['assay'] = 12  # 10x 3' v2
    adata_aligned.obs['nicheformer_split'] = 'train'
    
    logger.info("Creating NicheFormer dataset...")
    
    # Create dataset
    dataset = NicheformerDataset(
        adata=adata_aligned,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=1500,
        aux_tokens=30,
        chunk_size=1000,
        metadata_fields={'obs': ['modality', 'specie', 'assay']}
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info("Loading NicheFormer checkpoint...")
    
    # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        strict=False
    )
    model.eval()
    
    # Configure trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir=result_dir,
        precision=32,
    )
    
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
                layer=-1  # Last layer
            )
            embeddings.append(emb.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Store in adata
    adata.obsm["X_nicheformer"] = embeddings
    
    # Save result
    output_file = os.path.join(result_dir, "adata_nicheformer.h5ad")
    adata.write_h5ad(output_file)
    logger.info(f"Saved output to {output_file}")
    
    logger.info("✓ NicheFormer embedding extraction complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract NicheFormer embeddings')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference (default: 32)')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    extract_nicheformer(args.dataset, gpu_id=args.gpu, batch_size=args.batch_size)
    logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
