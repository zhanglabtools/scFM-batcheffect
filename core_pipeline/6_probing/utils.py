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
from datetime import datetime
import concurrent.futures
from multiprocessing import Pool
import functools

warnings.filterwarnings('ignore')

class CVDataSplitManager:
    """5折交叉验证数据划分管理器"""
    
    def __init__(self, save_dir="./cv_splits"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.split_file = os.path.join(save_dir, "cv_splits_5fold.pkl")
        self.config_file = os.path.join(save_dir, "cv_config.json")
        
        print(f"CV数据划分管理器初始化完成，保存目录: {self.save_dir}")
    
    def create_cv_splits(self, adata, target_key="study", n_splits=5, random_state=42, 
                        stratify_column=None, force_recreation=False):
        """创建5折交叉验证划分"""
        
        # 检查是否已有划分文件
        if os.path.exists(self.split_file) and not force_recreation:
            raise FileExistsError(
                f"CV划分文件已存在: {self.split_file}\n"
                f"如果要重新创建，请设置 force_recreation=True"
            )
        
        print(f"创建{n_splits}折交叉验证划分...")
        
        # 检查必要的列
        if 'cell_id' not in adata.obs.columns:
            raise ValueError("adata.obs中缺少'cell_id'列")
        
        # 提取数据
        cell_ids = adata.obs['cell_id'].values
        labels = adata.obs[target_key].values
        
        # 确定分层依据
        if stratify_column is None:
            stratify_labels = labels
            stratify_info = f"按{target_key}分层"
        elif stratify_column in adata.obs.columns:
            stratify_labels = adata.obs[stratify_column].values
            stratify_info = f"按{stratify_column}分层"
        else:
            print(f"警告: 未找到'{stratify_column}'列，将按{target_key}分层")
            stratify_labels = labels
            stratify_info = f"按{target_key}分层(fallback)"
        
        # 创建5折划分
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
            
            print(f"Fold {fold_idx}: 训练集 {len(train_cell_ids)}, 测试集 {len(test_cell_ids)}")
        
        # 添加元信息
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
        
        # 保存划分
        with open(self.split_file, 'wb') as f:
            pickle.dump(cv_splits, f)
        
        # 保存配置
        config = cv_splits['metadata'].copy()
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"CV划分已保存到: {self.split_file}")
        return cv_splits
    
    def load_cv_splits(self):
        """加载CV划分"""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"CV划分文件不存在: {self.split_file}")
        
        with open(self.split_file, 'rb') as f:
            cv_splits = pickle.load(f)
        
        n_splits = cv_splits['metadata']['n_splits']
        # print(f"CV划分已加载: {n_splits}折")
        return cv_splits
    
    def get_fold_indices(self, adata, fold_idx, cv_splits=None):
        """获取指定fold的训练/测试索引"""
        if cv_splits is None:
            cv_splits = self.load_cv_splits()
        
        fold_key = f'fold_{fold_idx}'
        if fold_key not in cv_splits:
            raise ValueError(f"不存在的fold: {fold_idx}")
        
        fold_data = cv_splits[fold_key]
        cell_ids = adata.obs['cell_id'].values
        cell_id_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        
        # 获取索引
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
    """支持5折交叉验证的线性探测分析器"""
    
    def __init__(self, save_dir, fold_idx):
        """
        初始化分析器
        
        Parameters:
        -----------
        save_dir : str
            模型保存目录（例如：./results/cellplm）
        fold_idx : int
            当前fold索引
        """
        self.fold_idx = fold_idx
        self.fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        os.makedirs(self.fold_dir, exist_ok=True)
        
        # 数据相关
        self.adata = None
        self.embedding_key = None
        self.target_key = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 预处理器和模型
        self.label_encoder = None
        self.scaler = None
        self.model = None
        
        # 文件路径
        self.data_split_path = os.path.join(self.fold_dir, "data_split.pkl")
        self.preprocessors_path = os.path.join(self.fold_dir, "preprocessors.pkl")
        self.model_path = os.path.join(self.fold_dir, "model.pkl")
        self.config_path = os.path.join(self.fold_dir, "config.json")
        # self.metrics_path = os.path.join(self.fold_dir, "metrics.json")
    
    def prepare_data(self, adata, embedding_key, target_key, train_indices, test_indices, 
                    normalize=True, force_recompute=False):
        """准备数据"""
        
        # 检查是否已有数据
        if not force_recompute and self._load_data_split() and self._load_preprocessors():
            # print(f"Fold {self.fold_idx}: 数据已从磁盘加载")
            return
        
        self.adata = adata
        self.embedding_key = embedding_key
        self.target_key = target_key
        
        # 提取数据
        X = adata.obsm[embedding_key]
        y = adata.obs[target_key].values
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 划分数据
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y_encoded[train_indices]
        self.y_test = y_encoded[test_indices]
        
        # 标准化
        if normalize:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        # 保存数据
        self._save_data_split()
        self._save_preprocessors()
        self._save_config(normalize=normalize)
        
        # print(f"Fold {self.fold_idx}: 数据准备完成, 训练集 {self.X_train.shape}, 测试集 {self.X_test.shape}")
    
    def train_and_evaluate(self, C=1.0, force_recompute=False):
        """训练并评估模型"""
        
        # 训练模型（可能从缓存加载）
        if not force_recompute and self._load_model():
            print(f"Fold {self.fold_idx}: 模型已从磁盘加载")
        else:
            print(f"Fold {self.fold_idx}: 开始训练...")
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1)
            self.model.fit(self.X_train, self.y_train)
            self._save_model()
            # print(f"Fold {self.fold_idx}: 模型训练完成并已保存")
        
        # 计算指标（不缓存）
        print(f"Fold {self.fold_idx}: 开始计算指标...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        overall_accuracy = accuracy_score(self.y_test, y_pred)
        
        # 计算每个类别的指标和macro-f1
        class_metrics, macro_f1 = self._calculate_class_metrics(cm)
        
        # 组织结果
        metrics = {
            'fold_idx': self.fold_idx,
            'overall_accuracy': overall_accuracy,
            'macro_f1': macro_f1,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        # print(f"Fold {self.fold_idx}: 评估完成，准确率 {overall_accuracy:.4f}, Macro-F1 {macro_f1:.4f}")
        return metrics

    def _calculate_class_metrics(self, cm):
        """计算每个类别的指标"""
        class_metrics = {}
        f1_scores = []  # 用于计算macro-f1
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            tp = cm[i, i]  # 该类别预测正确的样本数
            fp = cm[:, i].sum() - tp  # 假正例
            fn = cm[i, :].sum() - tp  # 假负例
            support = cm[i, :].sum()  # 该类别的总样本数
            
            # 类别准确率 = 该类别正确样本 / 该类别总样本
            class_accuracy = tp / support if support > 0 else 0
            
            # 计算precision和recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 计算F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
            class_metrics[class_name] = {
                'accuracy': class_accuracy,
                'f1_score': f1,
                'support': int(support)
            }
        
        # 计算macro-f1
        macro_f1 = np.mean(f1_scores) if f1_scores else 0
        
        return class_metrics, macro_f1
    
    
    # 保存/加载方法（与原来类似）
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


# 并行处理函数
def process_single_fold_model(args):
    """处理单个(模型, fold)组合的函数"""
    model_config, fold_idx, cv_split_manager, force_recompute = args
    
    try:
        import scanpy as sc
        import gc
        
        # 加载数据
        adata = sc.read_h5ad(model_config['adata_path'])
        
        # 获取fold的训练/测试索引
        cv_splits = cv_split_manager.load_cv_splits()
        train_indices, test_indices = cv_split_manager.get_fold_indices(adata, fold_idx, cv_splits)
        
        # 创建分析器
        analyzer = CVLinearProbingAnalyzer(model_config['save_path'], fold_idx)
        
        # 准备数据
        analyzer.prepare_data(
            adata=adata,
            embedding_key=model_config['embedding_key'],
            target_key=model_config['target_key'],
            train_indices=train_indices,
            test_indices=test_indices,
            normalize=True,
            force_recompute=force_recompute
        )
        
        # 训练和评估
        metrics = analyzer.train_and_evaluate(C=1.0, force_recompute=force_recompute)
        
        # 清理内存
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