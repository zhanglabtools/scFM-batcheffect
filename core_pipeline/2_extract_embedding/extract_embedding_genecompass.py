#!/usr/bin/env python3
"""
Extract GeneCompass embeddings

GeneCompass is a foundation model trained on 10M+ single-cell transcriptomes
with multi-modal prior knowledge (promoters, co-expression, gene families, PECA)

Input: {output_data_dir}/genecompass/patch*/ (HuggingFace dataset)
Output: {output_res_dir}/genecompass/cell_embeddings.npy
"""

import argparse
import os
import sys
import yaml
import glob
import random
import numpy as np
import torch
import logging
from datetime import datetime
from tqdm import tqdm
from datasets import load_from_disk

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


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CellEmbeddingExtractor:
    """Extract cell embeddings from GeneCompass model."""
    
    def __init__(self, checkpoint_path, prior_path, config, device='cuda:0'):
        """
        Initialize the cell embedding extractor.
        
        Args:
            checkpoint_path: Path to GeneCompass checkpoint
            prior_path: Base path for prior knowledge embeddings
            config: Configuration dictionary
            device: Device to use for inference
        """
        self.checkpoint_path = checkpoint_path
        self.prior_path = prior_path
        self.config = config
        self.device = device
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load GeneCompass model with prior knowledge."""
        logger.info("Loading GeneCompass model and prior knowledge...")
        
        # Add GeneCompass to path
        gc_code_path = self.config['model_paths']['genecompass']['code_path']
        if gc_code_path not in sys.path:
            sys.path.insert(0, gc_code_path)
        
        try:
            from genecompass import BertForMaskedLM
            from genecompass.utils import load_prior_embedding
        except ImportError as e:
            raise ImportError(f"Failed to import GeneCompass from {gc_code_path}: {e}")
        
        # Get resource paths from config
        gc_config = self.config['model_paths']['genecompass']
        resources = gc_config.get('resources', {})
        prior_base = gc_config.get('prior_knowledge_path', self.prior_path)
        
        # Build prior knowledge file paths
        prior_paths = {
            'promoter_human': os.path.join(prior_base, 'promoter_emb', 'human_emb_768.pickle'),
            'promoter_mouse': os.path.join(prior_base, 'promoter_emb', 'mouse_emb_768.pickle'),
            'coexp_human': os.path.join(prior_base, 'gene_co_express_emb', 'Human_dim_768_gene_28291_random.pickle'),
            'coexp_mouse': os.path.join(prior_base, 'gene_co_express_emb', 'Mouse_dim_768_gene_27444_random.pickle'),
            'family_human': os.path.join(prior_base, 'gene_family', 'Human_dim_768_gene_28291_random.pickle'),
            'family_mouse': os.path.join(prior_base, 'gene_family', 'Mouse_dim_768_gene_27934_random.pickle'),
            'peca_human': os.path.join(prior_base, 'PECA2vec', 'human_PECA_vec.pickle'),
            'peca_mouse': os.path.join(prior_base, 'PECA2vec', 'mouse_PECA_vec.pickle'),
            'id2name': resources.get('gene_id_name_path'),
            'token_dict': resources.get('token_path'),
            'homologous': os.path.join(prior_base, 'homologous_hm_token.pickle'),
        }
        
        # Validate paths
        for key, path in prior_paths.items():
            if path and not os.path.exists(path):
                logger.warning(f"Prior knowledge file not found: {key} at {path}")
        
        # Load prior embeddings
        logger.info("Loading prior embeddings...")
        try:
            out = load_prior_embedding(
                name2promoter_human_path=prior_paths['promoter_human'],
                name2promoter_mouse_path=prior_paths['promoter_mouse'],
                name2coexp_human_path=prior_paths['coexp_human'],
                name2coexp_mouse_path=prior_paths['coexp_mouse'],
                name2family_human_path=prior_paths['family_human'],
                name2family_mouse_path=prior_paths['family_mouse'],
                name2peca_human_path=prior_paths['peca_human'],
                name2peca_mouse_path=prior_paths['peca_mouse'],
                id2name_human_mouse_path=prior_paths['id2name'],
                token_dictionary_or_path=prior_paths['token_dict'],
                homologous_gene_path=prior_paths['homologous']
            )
            
            knowledges = {
                'promoter': out[0],
                'co_exp': out[1],
                'gene_family': out[2],
                'peca_grn': out[3],
                'homologous_gene_human2mouse': out[4],
            }
        except Exception as e:
            logger.error(f"Failed to load prior embeddings: {e}")
            raise
        
        # Load model
        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        try:
            self.model = BertForMaskedLM.from_pretrained(
                self.checkpoint_path,
                knowledges=knowledges,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.checkpoint_path}: {e}")
    
    def extract_from_dataset(self, dataset_path, output_path, batch_size=64, pooling_method='cls'):
        """
        Extract cell embeddings from preprocessed HuggingFace dataset.
        
        Args:
            dataset_path: Path to preprocessed dataset
            output_path: Path to save embeddings (.npy file)
            batch_size: Batch size for inference
            pooling_method: 'cls' or 'mean'
        """
        logger.info(f"Loading dataset from {dataset_path}...")
        data = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded: {len(data)} cells")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Extracting embeddings"):
                batch_end = min(i + batch_size, len(data))
                batch_data = data.select(range(i, batch_end))
                
                # Prepare batch tensors
                input_ids = torch.tensor(batch_data['input_ids']).to(self.device)
                values = torch.tensor(batch_data['values']).to(self.device)
                species = torch.tensor(batch_data['species']).to(self.device)
                
                # Forward pass
                outputs = self.model.bert.forward(
                    input_ids=input_ids,
                    values=values,
                    species=species
                )
                
                # Extract embeddings
                if pooling_method == 'cls':
                    batch_emb = outputs[0][:, 0, :].cpu().numpy()
                elif pooling_method == 'mean':
                    batch_emb = outputs[0][:, 1:, :].mean(dim=1).cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")
                
                embeddings.append(batch_emb)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings, allow_pickle=True)
        logger.info(f"Saved {embeddings.shape[0]} cell embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings


def extract_genecompass(dataset_name, config_path):
    """
    Extract GeneCompass embeddings.
    
    Args:
        dataset_name: Dataset name (e.g., 'limb', 'liver')
        config_path: Path to config.yaml
    """
    logger.info(f"Starting GeneCompass embedding extraction")
    
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
    gc_config = config['model_paths']['genecompass']
    checkpoint_path = gc_config.get('checkpoint_path')
    prior_path = gc_config.get('prior_knowledge_path')
    
    # Get GPU and batch size from config or arguments
    gpu_id = gc_config.get('gpu', 0)
    batch_size = gc_config.get('batch_size', 32)
    pooling_method = gc_config.get('pooling_method', 'cls')
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data dir: {output_data_dir}")
    logger.info(f"Result dir: {output_res_dir}")
    logger.info(f"GPU: {gpu_id}, Batch size: {batch_size}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(42)
    
    # Check input data exists
    genecompass_dir = os.path.join(output_data_dir, 'genecompass')
    if not os.path.exists(genecompass_dir):
        raise FileNotFoundError(f"GeneCompass directory not found at {genecompass_dir}")
    
    # Find dataset patches
    dataset_patches = sorted(glob.glob(os.path.join(genecompass_dir, 'patch*')))
    if not dataset_patches:
        raise FileNotFoundError(f"No patch directories found in {genecompass_dir}")
    
    logger.info(f"Found {len(dataset_patches)} dataset patch(es)")
    
    # Check model checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GeneCompass checkpoint not found at {checkpoint_path}")
    
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"Prior knowledge path not found at {prior_path}")
    
    # Initialize extractor
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    extractor = CellEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        prior_path=prior_path,
        config=config,
        device=device
    )
    
    # Create result directory
    result_dir = os.path.join(output_res_dir, 'genecompass')
    os.makedirs(result_dir, exist_ok=True)
    
    # Extract embeddings from all patches
    all_embeddings = []
    for patch_path in dataset_patches:
        patch_name = os.path.basename(patch_path)
        logger.info(f"Processing: {patch_name}")
        
        try:
            embeddings = extractor.extract_from_dataset(
                dataset_path=patch_path,
                output_path=os.path.join(result_dir, f"{patch_name}_embeddings.npy"),
                batch_size=batch_size,
                pooling_method=pooling_method
            )
            all_embeddings.append(embeddings)
        except Exception as e:
            logger.error(f"Error processing {patch_name}: {e}")
            raise
    
    # Concatenate embeddings from all patches
    if len(all_embeddings) > 1:
        final_embeddings = np.vstack(all_embeddings)
        output_file = os.path.join(result_dir, 'cell_embeddings.npy')
        np.save(output_file, final_embeddings, allow_pickle=True)
        logger.info(f"Saved concatenated embeddings: {final_embeddings.shape}")
    elif len(all_embeddings) == 1:
        output_file = os.path.join(result_dir, 'cell_embeddings.npy')
        np.save(output_file, all_embeddings[0], allow_pickle=True)
        logger.info(f"Saved embeddings: {all_embeddings[0].shape}")
    
    logger.info("GeneCompass embedding extraction complete")
    logger.info(f"Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract GeneCompass embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    try:
        extract_genecompass(
            dataset_name=args.dataset,
            config_path=args.config,
        )
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    main()