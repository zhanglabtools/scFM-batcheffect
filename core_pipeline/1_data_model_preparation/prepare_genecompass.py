#!/usr/bin/env python3
"""
Prepare GeneCompass model input from UCE h5ad

GeneCompass preprocessing pipeline (most complex):
1. Gene name → ID translation (via Gene_id_name_dict1.pickle)
2. Filter cells by z-score on gene counts/mito percentage
3. Gene ID filtering (intersection with gene_id_hpromoter.pickle)
4. Filter cells with <7 protein/miRNA genes
5. Filter by token dictionary (keep only genes in human_mouse_tokens.pickle)
6. Normalize by gene medians (human_gene_median_after_filter.pickle)
7. Log1p transformation
8. Rank value encoding with 2048 truncation
9. Output as HuggingFace Dataset

External Resources Required (from LCBERT):
- human_gene_median_after_filter.pickle: Gene median expression dict
- human_mouse_tokens.pickle: Token vocabulary dict
- Gene_id_name_dict1.pickle: Gene name → ID mapping
- gene_id_hpromoter.pickle: Filtered gene ID list
- {species}_protein_coding.txt: Protein-coding genes
- {species}_miRNA.txt: miRNA genes
- {species}_mitochondria.xlsx: Mitochondrial genes
"""

import argparse
import os
import logging
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tqdm
from scipy.stats import zscore
from datasets import Dataset, Features, Sequence, Value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add config loader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config_loader import load_config, get_model_config

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.warning(f"Could not load config.yaml, using hardcoded defaults: {e}")
    config = None

# Get GeneCompass resource paths from config
def get_genecompass_resources():
    if config:
        resources = config.get('model_paths', {}).get('genecompass', {}).get('resources', {})
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
        return {
            'dict_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle",
            'token_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/human_mouse_tokens.pickle",
            'gene_id_name_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/Gene_id_name_dict1.pickle",
            'gene_id_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/dict/gene_id_hpromoter.pickle",
            'protein_coding_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_protein_coding.txt",
            'mirna_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_miRNA.txt",
            'mitochondria_path': "/home/wanglinting/LCBERT/Download/GeneCompass/scdata/human_mitochondria.xlsx",
        }

# LCBERT resource paths (loaded from config or use defaults)
GENECOMPASS_RESOURCES = get_genecompass_resources()
DICT_PATH = GENECOMPASS_RESOURCES['dict_path']
TOKEN_PATH = GENECOMPASS_RESOURCES['token_path']
GENE_ID_NAME_PATH = GENECOMPASS_RESOURCES['gene_id_name_path']
GENE_ID_PATH = GENECOMPASS_RESOURCES['gene_id_path']
PROTEIN_CODING_PATH = GENECOMPASS_RESOURCES['protein_coding_path']
MIRNA_PATH = GENECOMPASS_RESOURCES['mirna_path']
MITOCHONDRIA_PATH = GENECOMPASS_RESOURCES['mitochondria_path']


def get_gene_list(species, file_list):
    """Extract protein-coding, miRNA, and mitochondria gene lists."""
    take_list = [f for f in file_list if f.split('/')[-1].split('_')[0] == species]
    
    protein_list = []
    miRNA_list = []
    mitochondria_list = []
    
    for f in take_list:
        if f.endswith('.txt'):
            name = f.split('/')[-1].split('_')[1]
            with open(f, 'r') as file:
                lines = [line.split()[0] for line in file]
                if name == 'protein':
                    protein_list.extend(lines)
                elif name == 'miRNA':
                    miRNA_list.extend(lines)
        elif f.endswith('.xlsx'):
            df = pd.read_excel(f)
            mitochondria_list = df.iloc[:, 1].tolist()
    
    return protein_list, miRNA_list, mitochondria_list


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


def normal_filter(adata, mito_list):
    """Filter cells by z-score on total counts and mitochondria percentage."""
    # Total counts per cell
    total_counts = np.array(adata.X.sum(axis=1)).squeeze()
    
    # Filter out cells with zero counts
    idx = total_counts > 0
    adata = adata[idx, :]
    total_counts = total_counts[idx]
    
    # Mitochondria counts
    gene_names = adata.var.index.tolist()
    mito_mask = [g in mito_list for g in gene_names]
    adata_mito = adata[:, mito_mask]
    mito_counts = np.array(adata_mito.X.sum(axis=1)).squeeze()
    mito_percentage = mito_counts / total_counts
    
    # Z-score filtering (keep cells within ±3σ)
    total_zscore = zscore(total_counts)
    mito_zscore = zscore(mito_percentage)
    keep_mask = (total_zscore > -3) & (total_zscore < 3) & (mito_zscore > -3) & (mito_zscore < 3)
    
    return adata[keep_mask, :]


def gene_number_filter(adata, gene_list):
    """Filter cells with <7 protein/miRNA genes."""
    indices = [g in gene_list for g in adata.var.index]
    adata_subset = adata[:, indices]
    
    # Count non-zero genes per cell
    if sp.issparse(adata_subset.X):
        nonzero_per_cell = adata_subset.X.getnnz(axis=1)
    else:
        nonzero_per_cell = np.count_nonzero(adata_subset.X, axis=1)
    
    keep_mask = nonzero_per_cell > 6
    return adata[keep_mask, :]


def normalize_by_median(adata, dict_path):
    """Normalize by dividing each gene by its median expression."""
    with open(dict_path, 'rb') as f:
        gene_median_dict = pickle.load(f)
    
    gene_list = adata.var.index.tolist()
    gene_medians = np.array([gene_median_dict.get(g, 1) for g in gene_list])
    
    X = adata.X
    if sp.issparse(X):
        # Sparse: divide columns by gene medians
        inv_medians = 1.0 / gene_medians
        D = sp.diags(inv_medians)
        X_normalized = X.dot(D).toarray()
    else:
        # Dense: broadcast division
        X_normalized = X / gene_medians
    
    adata.X = X_normalized
    return adata


def tokenize_cell(gene_vector, gene_ids, token_dict):
    """Convert expression vector to token IDs and values."""
    # Get non-zero positions
    nonzero_idx = np.nonzero(gene_vector)[0]
    
    # Sort by expression (descending)
    sorted_positions = np.argsort(-gene_vector[nonzero_idx])
    
    # Get gene IDs in sorted order
    sorted_genes = gene_ids[nonzero_idx][sorted_positions]
    
    # Map to tokens
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
        # Extract cell vector
        if is_sparse:
            row = X.getrow(idx).toarray().ravel()
        else:
            row = X[idx].ravel()
        
        # Tokenize
        tokens, vals = tokenize_cell(row, gene_ids, token_dict)
        
        # Truncate or pad to 2048
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


def prepare_genecompass(dataset_dir, species='human', patch_id=1):
    """
    Full GeneCompass preprocessing pipeline.
    
    Input: {dataset_dir}/uce/adata.h5ad (from UCE format, which has counts)
    Output: {dataset_dir}/genecompass/patch1/ (HuggingFace Dataset)
    """
    logger.info(f"Preparing GeneCompass input from {dataset_dir}")
    
    # Check input file
    input_file = os.path.join(dataset_dir, 'uce', 'adata.h5ad')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"UCE h5ad not found at {input_file}")
    
    # Check external resources
    required_files = [DICT_PATH, TOKEN_PATH, GENE_ID_NAME_PATH, GENE_ID_PATH]
    for fpath in required_files:
        if not os.path.exists(fpath):
            logger.warning(f"Missing external resource: {fpath}")
    
    # Load data
    adata = sc.read_h5ad(input_file)
    logger.info(f"Loaded data: {adata.shape}")
    
    # Load dictionaries
    with open(GENE_ID_NAME_PATH, 'rb') as f:
        dict_name_to_id = pickle.load(f)
    with open(TOKEN_PATH, 'rb') as f:
        token_dict = pickle.load(f)
    with open(GENE_ID_PATH, 'rb') as f:
        gene_id_list = pickle.load(f)
    
    # Get gene lists
    file_list = [PROTEIN_CODING_PATH, MIRNA_PATH, MITOCHONDRIA_PATH]
    protein_list, mirna_list, mito_list = get_gene_list(species, file_list)
    logger.info(f"Loaded {len(protein_list)} protein, {len(mirna_list)} miRNA, {len(mito_list)} mito genes")
    
    # Step 1: Map gene names to IDs
    logger.info("Step 1: Name → ID translation")
    gene_ids = id_name_match1(adata.var.index.tolist(), dict_name_to_id)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_ids
    adata = adata[:, ~(adata.var.index == 'delete')]
    logger.info(f"After name→ID: {adata.shape}")
    
    # Step 2: Filter by z-score (optional, commented in reference code)
    # adata = normal_filter(adata, mito_list)
    
    # Step 3: Gene ID filtering
    logger.info("Step 3: Gene ID filtering")
    adata = gene_id_filter(adata, gene_id_list)
    logger.info(f"After gene ID filter: {adata.shape}")
    
    # Step 4: Filter by gene count (optional, commented in reference)
    # adata = gene_number_filter(adata, protein_list + mirna_list)
    
    # Step 4+: Filter by token dictionary
    logger.info("Step 4+: Token dictionary filtering")
    gene_ids_filtered = id_token_match(adata.var.index.tolist(), token_dict)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_ids_filtered
    adata = adata[:, ~(adata.var.index == 'delete')]
    logger.info(f"After token filter: {adata.shape}")
    
    # Step 5: Normalize by median
    logger.info("Step 5: Normalization by gene medians")
    adata = normalize_by_median(adata, DICT_PATH)
    
    # Step 6: Log1p
    logger.info("Step 6: Log1p transformation")
    sc.pp.log1p(adata, base=2)
    
    # Step 7: Rank encoding
    logger.info("Step 7: Rank value tokenization")
    input_ids, lengths, values = rank_value(adata, token_dict)
    
    # Step 8: Create HuggingFace Dataset
    logger.info("Step 8: Create HuggingFace Dataset")
    hf_dataset = create_huggingface_dataset(species, lengths, input_ids, values, adata.n_obs)
    
    # Step 9: Save
    output_dir = os.path.join(dataset_dir, 'genecompass', f'patch{patch_id}')
    os.makedirs(output_dir, exist_ok=True)
    hf_dataset.save_to_disk(output_dir)
    logger.info(f"Saved HuggingFace Dataset to {output_dir}")
    
    # Save sorted lengths
    sorted_lengths = sorted(lengths)
    with open(os.path.join(output_dir, 'sorted_length.pickle'), 'wb') as f:
        pickle.dump(sorted_lengths, f)
    logger.info(f"Saved sorted lengths to {output_dir}/sorted_length.pickle")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GeneCompass model input')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset directory')
    parser.add_argument('--species', type=str, default='human', help='Species: human or mouse')
    parser.add_argument('--patch-id', type=int, default=1, help='Patch ID')
    
    args = parser.parse_args()
    prepare_genecompass(args.dataset, species=args.species, patch_id=args.patch_id)
