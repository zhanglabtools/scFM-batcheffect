#!/usr/bin/env python3
"""
Unified configuration loader for scFM pipeline.
Provides functions to load and access configuration from config.yaml

Usage:
    from config_loader import load_config, get_model_path, get_dataset_config
    
    config = load_config()
    model_path = get_model_path(config, 'geneformer', 'model_dir')
    dataset_cfg = get_dataset_config(config, 'limb')
"""

import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches in standard locations.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Search in standard locations
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
            '/home/wanglinting/scFM/Src/config.yaml',
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found in standard locations. "
                "Please provide explicit path via config_path parameter."
            )
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def get_model_config(config, model_name, default=None):
    """
    Get configuration for specific model.
    
    Args:
        config: Configuration dictionary
        model_name: Model name (geneformer, genecompass, etc.)
        default: Default configuration if model not found
    
    Returns:
        dict: Model configuration or default
    """
    model_cfg = config.get('model_paths', {}).get(model_name.lower())
    if model_cfg is None:
        if default is not None:
            logger.warning(f"Model configuration not found for '{model_name}', using default")
            return default
        raise KeyError(f"Model configuration not found for: {model_name}")
    return model_cfg


def get_model_path(config, model_name, key='model_dir', default=None):
    """
    Get specific path for a model.
    
    Args:
        config: Configuration dictionary
        model_name: Model name
        key: Path key (model_dir, token_dict, code_path, gene_list, etc.)
        default: Default value if key not found
    
    Returns:
        str: Path value or default
    """
    try:
        model_cfg = get_model_config(config, model_name)
        value = model_cfg.get(key)
        if value is None:
            if default is not None:
                logger.warning(f"Path key '{key}' not found for model '{model_name}', using default: {default}")
                return default
            raise KeyError(f"Path key '{key}' not found for model: {model_name}")
        return value
    except KeyError:
        if default is not None:
            logger.warning(f"Model '{model_name}' not found in config, using default: {default}")
            return default
        raise


def get_dataset_config(config, dataset_name):
    """
    Get configuration for specific dataset.
    
    Args:
        config: Configuration dictionary
        dataset_name: Dataset name (limb, liver, etc.)
    
    Returns:
        dict: Dataset configuration
    """
    dataset_cfg = config.get('datasets', {}).get(dataset_name)
    if dataset_cfg is None:
        raise KeyError(f"Dataset configuration not found for: {dataset_name}")
    return dataset_cfg


def get_code_path(config, code_name):
    """
    Get code repository path.
    
    Args:
        config: Configuration dictionary
        code_name: Code repository name (lcbert_base, geneformer_code, etc.)
    
    Returns:
        str: Path to code repository
    """
    code_path = config.get('code_paths', {}).get(code_name)
    if code_path is None:
        raise KeyError(f"Code path not found for: {code_name}")
    return code_path


def get_data_source(config, source_name):
    """
    Get data source path.
    
    Args:
        config: Configuration dictionary
        source_name: Data source name (cellxgene_base, hlca_base, etc.)
    
    Returns:
        str: Path to data source
    """
    source_path = config.get('data_sources', {}).get(source_name)
    if source_path is None:
        raise KeyError(f"Data source not found for: {source_name}")
    return source_path


def get_output_base(config):
    """Get output base directory."""
    return config.get('output_base', '/home/wanglinting/scFM/Data/Evaluation')


def get_model_param(config, model_name, param_key, default=None):
    """
    Get arbitrary parameter from model configuration.
    
    Args:
        config: Configuration dictionary
        model_name: Model name
        param_key: Parameter key to retrieve
        default: Default value if not found
    
    Returns:
        Parameter value or default
    """
    try:
        model_cfg = get_model_config(config, model_name, default={})
        value = model_cfg.get(param_key, default)
        return value
    except KeyError:
        return default


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    
    # Print some sample values
    print(f"GeneFormer model path: {get_model_path(config, 'geneformer')}")
    print(f"CELLxGENE base path: {get_data_source(config, 'cellxgene_base')}")
    print(f"Limb dataset output: {get_dataset_config(config, 'limb')['output_dir']}")
    print("✓ Configuration loaded successfully")
