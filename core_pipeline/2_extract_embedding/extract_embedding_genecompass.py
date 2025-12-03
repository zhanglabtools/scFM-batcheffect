#!/usr/bin/env python3
"""
Extract GeneCompass embeddings

GeneCompass is a foundation model trained on 10M+ single-cell transcriptomes
with multi-modal prior knowledge (promoters, co-expression, gene families, PECA)

Input: {dataset_dir}/genecompass/patch*/ (HuggingFace dataset from 0b_data_model_preparation)
Output: {dataset_dir}/../Result/{dataset_name}/genecompass/cell_embeddings.npy
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
import time
import argparse
import logging
from datetime import datetime

# Set timezone
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add config loader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config_loader import load_config, get_model_config, get_model_path

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.warning(f"Could not load config.yaml, using hardcoded defaults: {e}")
    config = None

# Add GeneCompass to path
def get_genecompass_code_path():
    if config:
        return get_model_path(config, 'genecompass', 'code_path',
               "/home/wanglinting/LCBERT/Code/model-genecompass")
    return "/home/wanglinting/LCBERT/Code/model-genecompass"

sys.path.append(get_genecompass_code_path())
from genecompass import BertForMaskedLM
from genecompass.utils import load_prior_embedding


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_genecompass_paths():
    """Get GeneCompass resource paths from config."""
    if config:
        genecompass_cfg = config.get('model_paths', {}).get('genecompass', {})
        resources = genecompass_cfg.get('resources', {})
        
        return {
            'dict_path': resources.get('dict_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle"),
            'token_path': resources.get('token_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_mouse_tokens.pickle"),
            'gene_id_name_path': resources.get('gene_id_name_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/Gene_id_name_dict1.pickle"),
            'gene_id_path': resources.get('gene_id_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/gene_id_hpromoter.pickle"),
            'protein_coding_path': resources.get('protein_coding_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_protein_coding.txt"),
            'mirna_path': resources.get('mirna_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_miRNA.txt"),
            'mitochondria_path': resources.get('mitochondria_path', "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_mitochondria.xlsx"),
        }
    else:
        # Fallback to hardcoded defaults
        return {
            'dict_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle",
            'token_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_mouse_tokens.pickle",
            'gene_id_name_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/Gene_id_name_dict1.pickle",
            'gene_id_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/gene_id_hpromoter.pickle",
            'protein_coding_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_protein_coding.txt",
            'mirna_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_miRNA.txt",
            'mitochondria_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_mitochondria.xlsx",
        }


class CellEmbeddingExtractor:
    """Extract cell embeddings from GeneCompass model."""
    
    def __init__(self, checkpoint_path, prior_path, device='cuda:0'):
        """
        Initialize the cell embedding extractor.
        
        Args:
            checkpoint_path: Path to GeneCompass checkpoint
            prior_path: Base path for prior knowledge embeddings
            device: Device to use for inference
        """
        self.checkpoint_path = checkpoint_path
        self.prior_path = prior_path
        self.device = device
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load GeneCompass model with prior knowledge."""
        logger.info("Loading prior knowledge embeddings...")
        
        # Get resource paths from config
        paths = get_genecompass_paths()
        
        # Prior knowledge file paths
        prior_paths = {
            'promoter_human': os.path.join(self.prior_path, 'promoter_emb', 'human_emb_768.pickle'),
            'promoter_mouse': os.path.join(self.prior_path, 'promoter_emb', 'mouse_emb_768.pickle'),
            'coexp_human': os.path.join(self.prior_path, 'gene_co_express_emb', 'Human_dim_768_gene_28291_random.pickle'),
            'coexp_mouse': os.path.join(self.prior_path, 'gene_co_express_emb', 'Mouse_dim_768_gene_27444_random.pickle'),
            'family_human': os.path.join(self.prior_path, 'gene_family', 'Human_dim_768_gene_28291_random.pickle'),
            'family_mouse': os.path.join(self.prior_path, 'gene_family', 'Mouse_dim_768_gene_27934_random.pickle'),
            'peca_human': os.path.join(self.prior_path, 'PECA2vec', 'human_PECA_vec.pickle'),
            'peca_mouse': os.path.join(self.prior_path, 'PECA2vec', 'mouse_PECA_vec.pickle'),
            'id2name': paths.get('gene_id_name_path', os.path.join(self.prior_path, 'gene_list', 'Gene_id_name_dict_human_mouse.pickle')),
            'token_dict': paths.get('token_path', os.path.join(self.prior_path, 'h&m_token1000W.pickle')),
            'homologous': os.path.join(self.prior_path, 'homologous_hm_token.pickle'),
        }
        
        # Load prior embeddings
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
        
        logger.info(f"Loading model from {self.checkpoint_path}...")
        self.model = BertForMaskedLM.from_pretrained(
            self.checkpoint_path,
            knowledges=knowledges,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def extract_from_dataset(self, dataset_path, output_path, batch_size=64, pooling_method='cls'):
        """
        Extract cell embeddings from preprocessed HuggingFace dataset.
        
        Args:
            dataset_path: Path to preprocessed dataset
            output_path: Path to save embeddings (.npy file)
            batch_size: Batch size for inference
            pooling_method: 'cls' or 'mean' - how to pool token embeddings
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
                
                # Extract embeddings based on pooling method
                if pooling_method == 'cls':
                    # Use [CLS] token embedding (first token)
                    batch_emb = outputs[0][:, 0, :].cpu().numpy()
                elif pooling_method == 'mean':
                    # Mean pooling over all tokens (excluding [CLS])
                    batch_emb = outputs[0][:, 1:, :].mean(dim=1).cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")
                
                embeddings.append(batch_emb)
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        logger.info(f"Saving embeddings to {output_path}...")
        np.save(output_path, embeddings, allow_pickle=True)
        logger.info(f"Saved {embeddings.shape[0]} cell embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings


def extract_genecompass(dataset_dir, gpu_id=0, batch_size=64):
    """
    Extract GeneCompass embeddings.
    
    Args:
        dataset_dir: Dataset directory containing 'genecompass/' subdirectory
        gpu_id: GPU device ID to use
        batch_size: Inference batch size
    """
    logger.info(f"Starting GeneCompass embedding extraction from {dataset_dir}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(42)
    
    # Check input data exists
    genecompass_dir = os.path.join(dataset_dir, 'genecompass')
    if not os.path.exists(genecompass_dir):
        raise FileNotFoundError(f"GeneCompass directory not found at {genecompass_dir}")
    
    # Find dataset patches
    import glob
    dataset_patches = sorted(glob.glob(os.path.join(genecompass_dir, 'patch*')))
    if not dataset_patches:
        raise FileNotFoundError(f"No patch directories found in {genecompass_dir}")
    
    logger.info(f"Found {len(dataset_patches)} dataset patch(es)")
    
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_dir)
    
    # Create output directory
    result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name, 'genecompass')
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Output directory: {result_dir}")
    
    # Model configuration
    checkpoint_path = "/home/wanglinting/LCBERT/Download/GeneCompass/pretrained_models/GeneCompass_Base"
    prior_path = "/home/wanglinting/LCBERT/Download/GeneCompass/prior_knowledge"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GeneCompass model not found at {checkpoint_path}")
    
    logger.info(f"Model path: {checkpoint_path}")
    
    # Initialize extractor
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    extractor = CellEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        prior_path=prior_path,
        device=device
    )
    
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
                pooling_method='cls'
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
    
    logger.info("✓ GeneCompass embedding extraction complete")
    logger.info(f"Results saved to {result_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract GeneCompass embeddings')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference (default: 64)')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    extract_genecompass(args.dataset, gpu_id=args.gpu, batch_size=args.batch_size)
    logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
