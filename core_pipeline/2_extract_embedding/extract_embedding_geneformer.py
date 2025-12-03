#!/usr/bin/env python3
"""
Extract GeneFormer embeddings using LCBERT's EmbExtractor

GeneFormer is a transformer-based model trained on 30M single-cell transcriptomes
Embeddings extracted from CLS token (or mean-pooled)

Input: {output_data_dir}/geneformer/*.dataset (tokenized dataset)
Output: {output_res_dir}/geneformer/*_emb.pt
"""

import argparse
import os
import sys
import yaml
import glob
import logging
from datetime import datetime

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


def extract_geneformer(dataset_name, config_path):
    """
    Extract GeneFormer embeddings.
    
    Args:
        dataset_name: Dataset name (e.g., 'limb', 'liver')
        config_path: Path to config.yaml
    """
    logger.info("Starting GeneFormer embedding extraction")
    
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
    gf_config = config['model_paths']['geneformer']
    model_path = gf_config['model_dir']
    token_dict_path = gf_config['token_dict']
    code_path = gf_config['code_path']
    gpu_id = gf_config.get('gpu', 0)
    batch_size = gf_config.get('batch_size', 32)
    nproc = gf_config.get('nproc', 8)
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data dir: {output_data_dir}")
    logger.info(f"Result dir: {output_res_dir}")
    logger.info(f"GPU: {gpu_id}, Batch size: {batch_size}, Processes: {nproc}")
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # Check input data exists
    dataset_dir_path = os.path.join(output_data_dir, 'geneformer')
    if not os.path.exists(dataset_dir_path):
        raise FileNotFoundError(f"GeneFormer dataset directory not found at {dataset_dir_path}")
    
    dataset_files = sorted(glob.glob(os.path.join(dataset_dir_path, '*.dataset')))
    if not dataset_files:
        raise FileNotFoundError(f"No .dataset files found in {dataset_dir_path}")
    
    logger.info(f"Found {len(dataset_files)} dataset file(s)")
    
    # Check model and token dict exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GeneFormer model not found at {model_path}")
    if not os.path.exists(token_dict_path):
        raise FileNotFoundError(f"Token dictionary not found at {token_dict_path}")
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Token dictionary: {token_dict_path}")
    
    # Add GeneFormer to path
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    
    try:
        from geneformer import EmbExtractor
    except ImportError as e:
        raise ImportError(f"Failed to import GeneFormer from {code_path}: {e}")
    
    # Initialize extractor
    logger.info("Initializing EmbExtractor...")
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        filter_data=None,
        max_ncells=None,
        emb_layer=-1,
        emb_label=['cell_id'],
        forward_batch_size=batch_size,
        nproc=nproc,
        token_dictionary_file=token_dict_path
    )
    
    # Create output directory
    result_dir = os.path.join(output_res_dir, 'geneformer')
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Output directory: {result_dir}")
    
    # Extract embeddings from all dataset files
    logger.info(f"Extracting embeddings from {len(dataset_files)} dataset file(s)...")
    for dataset_path in dataset_files:
        dataset_name_short = os.path.basename(dataset_path).replace('.dataset', '')
        logger.info(f"  Processing: {dataset_name_short}")
        
        try:
            embs = embex.extract_embs(
                model_directory=model_path,
                input_data_file=dataset_path,
                output_directory=result_dir,
                output_prefix=f"{dataset_name_short}_emb"
            )
            if hasattr(embs, 'shape'):
                logger.info(f"  ✓ Extracted embeddings: {embs.shape}")
            else:
                logger.info(f"  ✓ Embeddings extracted")
        except Exception as e:
            logger.error(f"Error processing {dataset_name_short}: {e}")
            raise
    
    logger.info("✓ GeneFormer embedding extraction complete")
    logger.info(f"Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract GeneFormer embeddings')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    logger.info(f"------ Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    
    try:
        extract_geneformer(args.dataset, args.config)
        logger.info(f"------ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        logger.error(f"------ Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ------")
        raise


if __name__ == '__main__':
    main()
