#!/usr/bin/env python3
"""
Extract GeneFormer embeddings using LCBERT's EmbExtractor

GeneFormer is a transformer-based model trained on 30M single-cell transcriptomes
Embeddings extracted from CLS token (or mean-pooled)

Input: {dataset_dir}/geneformer/*.dataset (tokenized dataset from 0b_data_model_preparation)
Output: {dataset_dir}/../Result/{dataset_name}/geneformer/*_emb.pt
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
import glob

# Set timezone
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add config loader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config_loader import load_config, get_model_path

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.warning(f"Could not load config.yaml, using hardcoded defaults: {e}")
    config = None


def get_model_configs():
    """Get GeneFormer model configuration from config.yaml."""
    if config is None:
        # Fallback to hardcoded defaults
        return {
            'model_path': "/home/wanglinting/LCBERT/Download/GeneformerV2/Geneformer-V2-316M",
            'token_dict_path': "/home/wanglinting/LCBERT/Download/GeneformerV2/geneformer/token_dictionary_gc104M.pkl",
            'code_path': "/home/wanglinting/LCBERT/Code/model-geneformer",
        }
    
    gf_cfg = config.get('model_paths', {}).get('geneformer', {})
    return {
        'model_path': gf_cfg.get('model_dir', "/home/wanglinting/LCBERT/Download/GeneformerV2/Geneformer-V2-316M"),
        'token_dict_path': gf_cfg.get('token_dict', "/home/wanglinting/LCBERT/Download/GeneformerV2/geneformer/token_dictionary_gc104M.pkl"),
        'code_path': gf_cfg.get('code_path', "/home/wanglinting/LCBERT/Code/model-geneformer"),
    }


# Add GeneFormer to path (using config)
model_cfg = get_model_configs()
sys.path.append(model_cfg['code_path'])
from geneformer import EmbExtractor


def extract_geneformer(dataset_dir, gpu_id=0, batch_size=32):
    """
    Extract GeneFormer embeddings.
    
    Args:
        dataset_dir: Dataset directory containing 'geneformer/' subdirectory with .dataset files
        gpu_id: GPU device ID to use (default: 0)
        batch_size: Inference batch size (default: 32)
    """
    logger.info(f"Starting GeneFormer embedding extraction from {dataset_dir}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # Check input data exists
    dataset_dir_path = os.path.join(dataset_dir, 'geneformer')
    if not os.path.exists(dataset_dir_path):
        raise FileNotFoundError(f"GeneFormer dataset directory not found at {dataset_dir_path}")
    
    dataset_files = glob.glob(os.path.join(dataset_dir_path, '*.dataset'))
    if not dataset_files:
        raise FileNotFoundError(f"No .dataset files found in {dataset_dir_path}")
    
    logger.info(f"Found {len(dataset_files)} dataset files")
    
    # Extract dataset name from path
    dataset_name = os.path.basename(dataset_dir)
    
    # Create output directory
    result_dir = os.path.join(dataset_dir, '..', 'Result', dataset_name, 'geneformer')
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Output directory: {result_dir}")
    
    # Get model configuration from config.yaml
    model_cfg = get_model_configs()
    model_path = model_cfg['model_path']
    token_dict_path = model_cfg['token_dict_path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GeneFormer model not found at {model_path}")
    if not os.path.exists(token_dict_path):
        raise FileNotFoundError(f"Token dictionary not found at {token_dict_path}")
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Token dictionary: {token_dict_path}")
    
    # Initialize extractor
    # Parameters:
    #   model_type: "Pretrained" for base model
    #   num_classes: 0 for pretrained (no classification head)
    #   emb_mode: "cls" for CLS token embedding
    #   emb_layer: -1 for second-to-last layer (layer before head)
    #   forward_batch_size: batch size for inference
    #   nproc: number of processes for data loading
    
    logger.info("Initializing EmbExtractor...")
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        filter_data=None,
        max_ncells=None,
        emb_layer=-1,  # Use second-to-last layer
        emb_label=['cell_id'],
        forward_batch_size=batch_size,
        nproc=8,  # Number of data loading processes
        token_dictionary_file=token_dict_path
    )
    
    # Extract embeddings from all dataset files
    logger.info(f"Extracting embeddings from {len(dataset_files)} dataset file(s)...")
    for dataset_path in dataset_files:
        dataset_name_short = os.path.basename(dataset_path).replace('.dataset', '')
        logger.info(f"Processing: {dataset_name_short}")
        
        try:
            embs = embex.extract_embs(
                model_directory=model_path,
                input_data_file=dataset_path,
                output_directory=result_dir,
                output_prefix=f"{dataset_name_short}_emb"
            )
            logger.info(f"✓ Extracted embeddings shape: {embs.shape if hasattr(embs, 'shape') else 'N/A'}")
        except Exception as e:
            logger.error(f"Error processing {dataset_name_short}: {e}")
            raise
    
    logger.info("✓ GeneFormer embedding extraction complete")
    logger.info(f"Results saved to {result_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract GeneFormer embeddings')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference (default: 32)')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    extract_geneformer(args.dataset, gpu_id=args.gpu, batch_size=args.batch_size)
    logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
