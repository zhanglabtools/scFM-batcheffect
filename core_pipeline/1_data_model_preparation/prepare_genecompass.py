#!/usr/bin/env python3
"""
Prepare GeneCompass model input from UCE h5ad

GeneCompass preprocessing pipeline:
1. Gene name → ID translation (via Gene_id_name_dict1.pickle)
2. Gene ID filtering (intersection with gene_id_hpromoter.pickle)
3. Filter by token dictionary (keep only genes in human_mouse_tokens.pickle)
4. Normalize by gene medians (human_gene_median_after_filter.pickle)
5. Log1p transformation
6. Rank value encoding with 2048 truncation
7. Output as HuggingFace Dataset
"""

import argparse
import os
import logging
import yaml
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tqdm
from datasets import Dataset, Features, Sequence, Value

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def id_name_match1(name_list, dict_mapping):
    """Map gene names to IDs using dictionary; mark unmapped as 'delete'."""
    return [dict_mapping.get(name, 'delete') for name in name_list]


def id_token_match(name_list, token_dict):
    """Keep only genes present in token dictionary; mark missing as 'delete'."""
    return [name if token_dict.get(name) is not None else 'delete' for name in name_list]


def gene_id_filter(adata, gene_id_list):
    """Filter genes to intersection with gene_id_list."""
    mask = adata.var.index.isin(gene_id_list)
    return adata[:, mask]


def normalize_by_median(adata, dict_path):
    """Normalize by dividing each gene by its median expression."""
    with open(dict_path, 'rb') as f:
        gene_median_dict = pickle.load(f)
    
    gene_list = adata.var.index.tolist()
    gene_medians = np.array([gene_median_dict.get(g, 1) for g in gene_list])
    
    X = adata.X
    if sp.issparse(X):
        inv_medians = 1.0 / gene_medians
        D = sp.diags(inv_medians)
        X_normalized = X.dot(D).toarray()
    else:
        X_normalized = X / gene_medians
    
    adata.X = X_normalized
    return adata


def tokenize_cell(gene_vector, gene_ids, token_dict):
    """Convert expression vector to token IDs and values."""
    nonzero_idx = np.nonzero(gene_vector)[0]
    sorted_positions = np.argsort(-gene_vector[nonzero_idx])
    sorted_genes = gene_ids[nonzero_idx][sorted_positions]
    
    tokens = [token_dict[g] for g in sorted_genes]
    values = gene_vector[nonzero_idx][sorted_positions].tolist()
    
    return tokens, values


def rank_value(adata, token_dict):
    """Encode cells as ranked token sequences (truncate to 2048)."""
    n_cells = adata.n_obs
    input_ids = np.zeros((n_cells, 2048), dtype=np.int32)
    values = np.zeros((n_cells, 2048), dtype=np.float32)
    lengths = []
    
    gene_ids = np.array(adata.var.index.tolist())
    X = adata.X
    is_sparse = sp.issparse(X)
    
    for idx in tqdm.tqdm(range(n_cells), desc="Tokenizing cells"):
        if is_sparse:
            row = X.getrow(idx).toarray().ravel()
        else:
            row = X[idx].ravel()
        
        tokens, vals = tokenize_cell(row, gene_ids, token_dict)
        
        L = len(tokens)
        if L > 2048:
            input_ids[idx] = tokens[:2048]
            values[idx] = vals[:2048]
            lengths.append(2048)
        else:
            input_ids[idx, :L] = tokens
            values[idx, :L] = vals
            lengths.append(L)
    
    return input_ids, lengths, values


def create_huggingface_dataset(species_str, lengths, input_ids, values, n_cells):
    """Convert to HuggingFace Dataset format."""
    species_int = 0 if species_str == 'human' else 1
    species_list = [[species_int] for _ in range(n_cells)]
    lengths = [[l] for l in lengths]
    
    data_dict = {
        'input_ids': input_ids,
        'values': values,
        'length': lengths,
        'species': species_list
    }
    
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'values': Sequence(feature=Value(dtype='float32')),
        'length': Sequence(feature=Value(dtype='int16')),
        'species': Sequence(feature=Value(dtype='int16')),
    })
    
    return Dataset.from_dict(data_dict, features=features)


def prepare_genecompass(dataset_config, model_config, species='human', patch_id=1):
    """
    Full GeneCompass preprocessing pipeline.
    
    Input: {output_data_dir}/uce/adata.h5ad
    Output: {output_data_dir}/genecompass/patch{patch_id}/ (HuggingFace Dataset)
    """
    logger.info(f"Preparing GeneCompass input for {species}...")
    
    # Get output directory
    output_dir = dataset_config['output_data_dir']
    input_file = os.path.join(output_dir, 'uce', 'adata.h5ad')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"UCE h5ad not found at {input_file}")
    
    # Get resource paths from model_config
    gc_resources = model_config['genecompass'].get('resources', {})
    
    dict_path = gc_resources.get('dict_path')
    token_path = gc_resources.get('token_path')
    gene_id_name_path = gc_resources.get('gene_id_name_path')
    gene_id_path = gc_resources.get('gene_id_path')
    
    # Check resources exist
    for fpath in [dict_path, token_path, gene_id_name_path, gene_id_path]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing external resource: {fpath}")
    
    # Load data
    adata = sc.read_h5ad(input_file)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Load dictionaries
    with open(gene_id_name_path, 'rb') as f:
        dict_name_to_id = pickle.load(f)
    with open(token_path, 'rb') as f:
        token_dict = pickle.load(f)
    with open(gene_id_path, 'rb') as f:
        gene_id_list = pickle.load(f)
    
    logger.info("Loaded dicts and token vocabulary")
    
    # Step 1: Name → ID translation
    logger.info("Step 1: Name → ID translation")
    gene_ids = id_name_match1(adata.var.index.tolist(), dict_name_to_id)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_ids
    adata = adata[:, ~(adata.var.index == 'delete')]
    logger.info(f"After name→ID: {adata.shape}")
    
    # Step 2: Gene ID filtering
    logger.info("Step 2: Gene ID filtering")
    adata = gene_id_filter(adata, gene_id_list)
    logger.info(f"After gene ID filter: {adata.shape}")
    
    # Step 3: Filter by token dictionary
    logger.info("Step 3: Token dictionary filtering")
    gene_ids_filtered = id_token_match(adata.var.index.tolist(), token_dict)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_ids_filtered
    adata = adata[:, ~(adata.var.index == 'delete')]
    logger.info(f"After token filter: {adata.shape}")
    
    # Step 4: Normalize by median
    logger.info("Step 4: Normalization by gene medians")
    adata = normalize_by_median(adata, dict_path)
    
    # Step 5: Log1p
    logger.info("Step 5: Log1p transformation")
    sc.pp.log1p(adata, base=2)
    
    # Step 6: Rank encoding
    logger.info("Step 6: Rank value tokenization")
    input_ids, lengths, values = rank_value(adata, token_dict)
    
    # Step 7: Create HuggingFace Dataset
    logger.info("Step 7: Create HuggingFace Dataset")
    hf_dataset = create_huggingface_dataset(species, lengths, input_ids, values, adata.n_obs)
    
    # Step 8: Save
    output_subdir = os.path.join(output_dir, 'genecompass', f'patch{patch_id}')
    os.makedirs(output_subdir, exist_ok=True)
    hf_dataset.save_to_disk(output_subdir)
    logger.info(f"Saved HuggingFace Dataset to {output_subdir}")
    
    # Save sorted lengths
    sorted_lengths = sorted(lengths)
    with open(os.path.join(output_subdir, 'sorted_length.pickle'), 'wb') as f:
        pickle.dump(sorted_lengths, f)
    logger.info(f"Saved sorted lengths")


def main():
    parser = argparse.ArgumentParser(description='Prepare GeneCompass model input')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['limb', 'liver', 'Immune', 'HLCA_assay', 'HLCA_disease', 'HLCA_sn'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml file')
    parser.add_argument('--species', type=str, default='human',
                        choices=['human', 'mouse'],
                        help='Species: human or mouse')
    parser.add_argument('--patch-id', type=int, default=1,
                        help='Patch ID')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract dataset-specific configuration and model configuration
    dataset_config = config['datasets'][args.dataset]
    model_config = config['model_paths']
    logger.info(f"Loading dataset configuration for: {args.dataset}")
    
    # Prepare GeneCompass input
    prepare_genecompass(dataset_config, model_config, species=args.species, patch_id=args.patch_id)


if __name__ == '__main__':
    main()