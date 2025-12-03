# -*- coding: utf-8 -*-
# Create time: 2025-10-30
# Update time: 2025-10-30

# conda activate benchmark
# cd /home/wanglinting/scFM/Code/probing
# nohup python -u probing_data_spilt.py > probing_data_spilt.log 2>&1 & 

import os
import gc
import time
from datetime import datetime
import scanpy as sc
from utils_cv import CVDataSplitManager

os.environ['TZ'] = 'Asia/Shanghai'  # 设置为东八区
time.tzset()


# =============================================================================
# 功能函数
# =============================================================================

def create_cv_splits():
    """创建5折交叉验证数据划分"""
    
    print(f"\n开始创建5折交叉验证数据划分: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参考模型: {REFERENCE_MODEL['name']}")
    print(f"折数: {SPLIT_CONFIG['n_splits']}")
    print(f"随机种子: {SPLIT_CONFIG['random_state']}")
    print(f"分层列: {SPLIT_CONFIG['stratify_column']}")
    
    # 创建CV数据划分管理器
    split_manager = CVDataSplitManager(SPLIT_CONFIG['split_save_dir'])
    
    # 检查是否已有划分
    if os.path.exists(split_manager.split_file) and not FORCE_RECREATION:
        print(f"CV划分文件已存在，如需重新创建请设置 FORCE_RECREATION = True")
        cv_splits = split_manager.load_cv_splits()
        return
    
    # 加载参考数据
    print(f"加载数据: {REFERENCE_MODEL['adata_path']}")
    adata = sc.read_h5ad(REFERENCE_MODEL['adata_path'])
    print(f"数据形状: {adata.shape}")
    
    # 检查必要的列
    required_columns = ['cell_id', REFERENCE_MODEL['target_key']]
    if SPLIT_CONFIG['stratify_column']:
        required_columns.append(SPLIT_CONFIG['stratify_column'])
    
    for col in required_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 显示类别信息
    n_classes = len(adata.obs[REFERENCE_MODEL['target_key']].unique())
    print(f"目标标签类别数: {n_classes}")
    
    # 创建CV数据划分
    cv_splits = split_manager.create_cv_splits(
        adata=adata,
        target_key=REFERENCE_MODEL['target_key'],
        n_splits=SPLIT_CONFIG['n_splits'],
        random_state=SPLIT_CONFIG['random_state'],
        stratify_column=SPLIT_CONFIG['stratify_column'],
        force_recreation=FORCE_RECREATION
    )
    
    print(f"CV数据划分完成!")
    print(f"划分文件: {split_manager.split_file}")

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "Immune_batch"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/Immune/probing/batch/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'tissue'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/Immune/pca/Embeddings_pca.h5ad',
    'target_key': 'tissue'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "Immune_celltype"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/Immune/probing/celltype/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'cell_type_level_2'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/Immune/pca/Embeddings_pca.h5ad',
    'target_key': 'cell_type_level_2'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_assay_batch"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_assay/probing/batch/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'assay'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay/pca/Embeddings_pca.h5ad',
    'target_key': 'assay'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_disease_batch"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_disease/probing/batch/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'disease'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease/pca/Embeddings_pca.h5ad',
    'target_key': 'disease'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_sn_batch"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_sn/probing/batch/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'suspension_type'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn/pca/Embeddings_pca.h5ad',
    'target_key': 'suspension_type'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_assay_celltype"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_assay/probing/celltype/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'final_annotation'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay/pca/Embeddings_pca.h5ad',
    'target_key': 'final_annotation'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_disease_celltype"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_disease/probing/celltype/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'final_annotation'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease/pca/Embeddings_pca.h5ad',
    'target_key': 'final_annotation'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION

# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "HLCA_sn_celltype"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_sn/probing/celltype/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'final_annotation'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn/pca/Embeddings_pca.h5ad',
    'target_key': 'final_annotation'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION


# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "limb_batch"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/limb/probing/batch/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'organism'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/limb/pca/Embeddings_pca.h5ad',
    'target_key': 'organism'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION


# =============================================================================
# 配置参数
# =============================================================================

# 需要不同数据集有共同的cell_id列，用于唯一标识细胞

analysis_name = "limb_celltype"

# 数据划分配置
SPLIT_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/limb/probing/celltype/data_splits',
    'n_splits': 5,
    'random_state': 123,
    'stratify_column': 'cell_type'
}

# 参考模型配置（用于创建统一划分）
REFERENCE_MODEL = {
    'name': 'PCA',
    'adata_path': '/home/wanglinting/scFM/Result/limb/pca/Embeddings_pca.h5ad',
    'target_key': 'cell_type'
}

# 是否强制重新创建划分
FORCE_RECREATION = False

print(f"\n{'='*40} {analysis_name} probing数据划分开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
create_cv_splits()
print(f"\n{'='*40} {analysis_name} probing数据划分完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del REFERENCE_MODEL, SPLIT_CONFIG, analysis_name, FORCE_RECREATION


