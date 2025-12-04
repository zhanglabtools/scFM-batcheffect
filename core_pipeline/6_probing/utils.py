# -*- coding: utf-8 -*-
# Create time: 2025-09-16
# Update time: 2025-09-17

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import pickle
import json
import os
import logging
from datetime import datetime
import concurrent.futures
from multiprocessing import Pool
import functools

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CVDataSplitManager:
    """5-fold cross-validation data split manager"""
    
    def __init__(self, save_dir="./cv_splits"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.split_file = os.path.join(save_dir, "cv_splits_5fold.pkl")
        self.config_file = os.path.join(save_dir, "cv_config.json")
        
        logger.info(f"CV split manager initialized. Save directory: {self.save_dir}")
    
    def create_cv_splits(self, adata, target_key="study", n_splits=5, random_state=42, 
                        stratify_column=None, force_recreation=False):
        """Create n-fold cross-validation splits"""
        
        # Check if split file already exists
        if os.path.exists(self.split_file) and not force_recreation:
            raise FileExistsError(
                f"CV split file already exists: {self.split_file}\n"
                f"Set force_recreation=True to recreate it"
            )
        
        logger.info(f"Creating {n_splits}-fold cross-validation splits...")
        
        # Check required columns
        if 'cell_id' not in adata.obs.columns:
            raise ValueError("Missing 'cell_id' column in adata.obs")
        
        # Extract data
        cell_ids = adata.obs['cell_id'].values
        labels = adata.obs[target_key].values
        
        # Determine stratification basis
        if stratify_column is None:
            stratify_labels = labels
            stratify_info = f"Stratified by {target_key}"
        elif stratify_column in adata.obs.columns:
            stratify_labels = adata.obs[stratify_column].values
            stratify_info = f"Stratified by {stratify_column}"
        else:
            logger.warning(f"Column '{stratify_column}' not found, stratifying by {target_key}")
            stratify_labels = labels
            stratify_info = f"Stratified by {target_key} (fallback)"
        
        # Create n-fold splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = {}
        
        for fold_idx, (train_indices, test_indices) in enumerate(skf.split(cell_ids, stratify_labels)):
            train_cell_ids = cell_ids[train_indices].tolist()
            test_cell_ids = cell_ids[test_indices].tolist()
            
            cv_splits[f'fold_{fold_idx}'] = {
                'train_cell_ids': train_cell_ids,
                'test_cell_ids': test_cell_ids,
                'train_indices': train_indices.tolist(),
                'test_indices': test_indices.tolist()
            }
            
            logger.info(f"Fold {fold_idx}: train={len(train_cell_ids)}, test={len(test_cell_ids)}")
        
        # Add metadata
        cv_splits['metadata'] = {
            'n_splits': n_splits,
            'total_cells': len(cell_ids),
            'n_classes': len(set(labels)),
            'class_names': sorted(list(set(labels))),
            'random_state': random_state,
            'target_key': target_key,
            'stratify_column': stratify_column,
            'stratify_info': stratify_info,
            'created_time': datetime.now().isoformat()
        }
        
        # Save splits
        with open(self.split_file, 'wb') as f:
            pickle.dump(cv_splits, f)
        
        # Save config
        config = cv_splits['metadata'].copy()
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"CV splits saved to: {self.split_file}")
        return cv_splits
    
    def load_cv_splits(self):
        """Load CV splits"""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"CV split file not found: {self.split_file}")
        
        with open(self.split_file, 'rb') as f:
            cv_splits = pickle.load(f)
        
        n_splits = cv_splits['metadata']['n_splits']
        # logger.info(f"CV splits loaded: {n_splits}-fold")
        return cv_splits
    
    def get_fold_indices(self, adata, fold_idx, cv_splits=None):
        """Get train/test indices for specified fold"""
        if cv_splits is None:
            cv_splits = self.load_cv_splits()
        
        fold_key = f'fold_{fold_idx}'
        if fold_key not in cv_splits:
            raise ValueError(f"Invalid fold: {fold_idx}")
        
        fold_data = cv_splits[fold_key]
        cell_ids = adata.obs['cell_id'].values
        cell_id_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        
        # Get indices
        train_indices = []
        test_indices = []
        
        for cell_id in fold_data['train_cell_ids']:
            if cell_id in cell_id_to_index:
                train_indices.append(cell_id_to_index[cell_id])
        
        for cell_id in fold_data['test_cell_ids']:
            if cell_id in cell_id_to_index:
                test_indices.append(cell_id_to_index[cell_id])
        
        return np.array(train_indices), np.array(test_indices)


class CVLinearProbingAnalyzer:
    """Linear probing analyzer with 5-fold cross-validation support"""
    
    def __init__(self, save_dir, fold_idx):
        """
        Initialize analyzer
        
        Parameters:
        -----------
        save_dir : str
            Model save directory (e.g., ./results/cellplm)
        fold_idx : int
            Current fold index
        """
        self.fold_idx = fold_idx
        self.fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        os.makedirs(self.fold_dir, exist_ok=True)
        
        # Data-related attributes
        self.adata = None
        self.embedding_key = None
        self.target_key = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Preprocessors and model
        self.label_encoder = None
        self.scaler = None
        self.model = None
        
        # File paths
        self.data_split_path = os.path.join(self.fold_dir, "data_split.pkl")
        self.preprocessors_path = os.path.join(self.fold_dir, "preprocessors.pkl")
        self.model_path = os.path.join(self.fold_dir, "model.pkl")
        self.config_path = os.path.join(self.fold_dir, "config.json")
        # self.metrics_path = os.path.join(self.fold_dir, "metrics.json")
    
    def prepare_data(self, adata, embedding_key, target_key, train_indices, test_indices, 
                    normalize=True, force_recompute=False):
        """Prepare data for training"""
        
        # Check if data already exists
        if not force_recompute and self._load_data_split() and self._load_preprocessors():
            # logger.info(f"Fold {self.fold_idx}: Data loaded from disk")
            return
        
        self.adata = adata
        self.embedding_key = embedding_key
        self.target_key = target_key
        
        # Extract data
        X = adata.obsm[embedding_key]
        y = adata.obs[target_key].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y_encoded[train_indices]
        self.y_test = y_encoded[test_indices]
        
        # Normalize
        if normalize:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        # Save data
        self._save_data_split()
        self._save_preprocessors()
        self._save_config(normalize=normalize)
        
        # logger.info(f"Fold {self.fold_idx}: Data prepared. Train: {self.X_train.shape}, Test: {self.X_test.shape}")
    
    def train_and_evaluate(self, C=1.0, force_recompute=False):
        """Train and evaluate model"""
        
        # Train model (may load from cache)
        if not force_recompute and self._load_model():
            logger.info(f"Fold {self.fold_idx}: Model loaded from disk")
        else:
            logger.info(f"Fold {self.fold_idx}: Starting training...")
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1)
            self.model.fit(self.X_train, self.y_train)
            self._save_model()
            # logger.info(f"Fold {self.fold_idx}: Model training completed and saved")
        
        # Compute metrics (not cached)
        logger.info(f"Fold {self.fold_idx}: Computing metrics...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        overall_accuracy = accuracy_score(self.y_test, y_pred)
        
        # Compute metrics per class and macro-f1
        class_metrics, macro_f1 = self._calculate_class_metrics(cm)
        
        # Organize results
        metrics = {
            'fold_idx': self.fold_idx,
            'overall_accuracy': overall_accuracy,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        # logger.info(f"Fold {self.fold_idx}: Evaluation complete. Accuracy: {overall_accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
        return metrics

    def _calculate_class_metrics(self, cm):
        """Calculate metrics for each class"""
        class_metrics = {}
        f1_scores = []  # For calculating macro-f1
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            tp = cm[i, i]  # True positives for this class
            fp = cm[:, i].sum() - tp  # False positives
            fn = cm[i, :].sum() - tp  # False negatives
            support = cm[i, :].sum()  # Total samples in this class
            
            # Class accuracy = correct predictions / total samples for this class
            class_accuracy = tp / support if support > 0 else 0
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
            class_metrics[class_name] = {
                'accuracy': class_accuracy,
                'f1_score': f1,
                'support': int(support)
            }
        
        # Calculate macro-f1
        macro_f1 = np.mean(f1_scores) if f1_scores else 0
        
        return class_metrics, macro_f1
    
    
    # Save/load methods
    def _save_data_split(self):
        split_data = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
        with open(self.data_split_path, 'wb') as f:
            pickle.dump(split_data, f)
    
    def _load_data_split(self):
        if not os.path.exists(self.data_split_path):
            return False
        try:
            with open(self.data_split_path, 'rb') as f:
                split_data = pickle.load(f)
            self.X_train = split_data['X_train']
            self.X_test = split_data['X_test']
            self.y_train = split_data['y_train']
            self.y_test = split_data['y_test']
            return True
        except:
            return False
    
    def _save_preprocessors(self):
        preprocessors = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        with open(self.preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
    
    def _load_preprocessors(self):
        if not os.path.exists(self.preprocessors_path):
            return False
        try:
            with open(self.preprocessors_path, 'rb') as f:
                preprocessors = pickle.load(f)
            self.label_encoder = preprocessors['label_encoder']
            self.scaler = preprocessors['scaler']
            return True
        except:
            return False
    
    def _save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            return False
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except:
            return False
    
    def _save_config(self, **kwargs):
        config = {
            'fold_idx': self.fold_idx,
            'embedding_key': self.embedding_key,
            'target_key': self.target_key,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)


# Parallel processing function
def process_single_fold_model(args):
    """Process single (model, fold) combination"""
    model_config, fold_idx, cv_split_manager, force_recompute = args
    
    try:
        import scanpy as sc
        import gc
        
        # Load data
        adata = sc.read_h5ad(model_config['adata_path'])
        
        # Get train/test indices for fold
        cv_splits = cv_split_manager.load_cv_splits()
        train_indices, test_indices = cv_split_manager.get_fold_indices(adata, fold_idx, cv_splits)
        
        # Create analyzer
        analyzer = CVLinearProbingAnalyzer(model_config['save_path'], fold_idx)
        
        # Prepare data
        analyzer.prepare_data(
            adata=adata,
            embedding_key=model_config['embedding_key'],
            target_key=model_config['target_key'],
            train_indices=train_indices,
            test_indices=test_indices,
            normalize=True,
            force_recompute=force_recompute
        )
        
        # Train and evaluate
        metrics = analyzer.train_and_evaluate(C=1.0, force_recompute=force_recompute)
        
        # Clean up memory
        del adata
        del analyzer
        gc.collect()
        
        return {
            'model_name': model_config['name'],
            'fold_idx': fold_idx,
            'status': 'success',
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'model_name': model_config['name'],
            'fold_idx': fold_idx,
            'status': 'failed',
            'error': str(e)
        }
