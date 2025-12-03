# -*- coding: utf-8 -*-
# Create time: 2025-10-10
# Update time: 2025-11-17

# conda activate benchmark
# cd /home/wanglinting/scFM/Code/probing
# nohup python -u probing_analysis_batch_normalize.py > probing_analysis_batch_normalize.log 2>&1 & echo $! > probing_analysis_batch_normalize.pid

import os
import time
import gc
import json
import pandas as pd
from datetime import datetime
import concurrent.futures
from utils_cv import CVDataSplitManager, process_single_fold_model

# 设置为东八区
os.environ['TZ'] = 'Asia/Shanghai'  
time.tzset()

# =============================================================================
# 配置函数
# =============================================================================

def run_cv_probing_parallel(cv_config=None, models_config=None, results_path=None):
    """
    并行运行5折交叉验证线性探测
    
    参数:
        cv_config (dict): CV配置参数
        models_config (list): 模型配置列表
        results_path (str): 结果保存路径
    
    返回:
        list: 所有任务的结果列表
    """
    if cv_config is None:
        print("❌ cv_config不能为空")
        return None

    if models_config is None:
        print("❌ models_config不能为空")
        return None

    if results_path is None:
        print("❌ results_path不能为空")
        return None

    print(f"模型数量: {len(models_config)}")
    print(f"折数: {cv_config['n_splits']}")
    print(f"最大并行数: {cv_config['max_workers']}")
    
    # 加载CV数据划分管理器
    cv_split_manager = CVDataSplitManager(cv_config['split_save_dir'])
    
    # 验证CV划分是否存在
    try:
        cv_split_manager.load_cv_splits()
        print(f"使用已有的CV数据划分")
    except FileNotFoundError:
        print(f"未找到CV数据划分文件，请先运行 create_splits_cv.py")
        return None
    
    # 创建所有任务
    tasks = []
    for model_config in models_config:
        for fold_idx in range(cv_config['n_splits']):
            tasks.append((model_config, fold_idx, cv_split_manager, cv_config['force_recompute']))
    
    print(f"总任务数: {len(tasks)}")
    
    # 并行执行所有任务
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cv_config['max_workers']) as executor:
        # print(f"启动并行执行...")
        
        # 提交所有任务
        futures = []
        for i, task in enumerate(tasks):
            future = executor.submit(process_single_fold_model, task)
            futures.append((i, future))
        
        # 收集结果
        completed = 0
        for i, future in futures:
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                status = "✅" if result['status'] == 'success' else "❌"
                print(f"{status} [{completed}/{len(tasks)}] {result['model_name']} Fold {result['fold_idx']}")
                
            except Exception as e:
                print(f"❌ 任务 {i} 执行失败: {e}")
                # 添加失败记录
                task = tasks[i]
                all_results.append({
                    'model_name': task[0]['name'],
                    'fold_idx': task[1],
                    'status': 'failed',
                    'error': str(e)
                })
    
    print(f"并行执行完成，开始汇总结果...")
    
    # 汇总结果
    summarize_and_save_results(all_results, results_path, cv_config)
    
    return None

def summarize_and_save_results(all_results, results_path, cv_config):
    """
    汇总并保存CV结果
    
    参数:
        all_results (list): 所有任务的结果列表
        results_path (str): 结果保存路径
        cv_config (dict): CV配置参数
    """
        
    # 组织结果数据
    cv_data = []
    
    for result in all_results:
        if result['status'] == 'success':
            model_name = result['model_name']
            fold_idx = result['fold_idx']
            metrics = result['metrics']
            
            # 添加整体准确率和macro-f1
            cv_data.append({
                'model_name': model_name,
                'fold': fold_idx,
                'dataset': 'overall',
                'accuracy': metrics['overall_accuracy'],
                'f1_score': metrics['macro_f1'],  # 整体用macro-f1
                'support': sum([m['support'] for m in metrics['class_metrics'].values()])
            })
            
            # 添加每个类别的准确率和f1
            for class_name, class_metric in metrics['class_metrics'].items():
                cv_data.append({
                    'model_name': model_name,
                    'fold': fold_idx,
                    'dataset': class_name,
                    'accuracy': class_metric['accuracy'],
                    'f1_score': class_metric['f1_score'],  # 类别用单独的f1
                    'support': class_metric['support']
                })
    
    # 转换为DataFrame
    df = pd.DataFrame(cv_data)
    
    if len(df) > 0:
        # 计算统计汇总
        summary_data = []
        
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            
            for dataset in model_df['dataset'].unique():
                dataset_df = model_df[model_df['dataset'] == dataset]
                
                if len(dataset_df) >= cv_config['n_splits']:  # 确保有完整的fold数据
                    # 计算均值
                    summary_data.append({
                        'model_name': model_name,
                        'fold': 'mean',
                        'dataset': dataset,
                        'accuracy': dataset_df['accuracy'].mean(),
                        'f1_score': dataset_df['f1_score'].mean(),
                        'support': int(dataset_df['support'].mean())
                    })
                    
                    # 计算标准差
                    summary_data.append({
                        'model_name': model_name,
                        'fold': 'std',
                        'dataset': dataset,
                        'accuracy': dataset_df['accuracy'].std(),
                        'f1_score': dataset_df['f1_score'].std(),
                        'support': int(dataset_df['support'].std())
                    })
        
        # 添加汇总数据
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # 排序
        final_df = final_df.sort_values(['model_name', 'dataset', 'fold']).reset_index(drop=True)
        
        # 保存结果
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        final_df.to_csv(results_path, index=False, float_format='%.6f')
        
        print(f"CV结果已保存到: {results_path}")
        
        # 打印简要汇总
        print_brief_summary(final_df)
    
    else:
        print("❌ 没有成功的结果可以汇总")

def print_brief_summary(df):
    """打印简要结果汇总"""
    
    print(f"\n{'='*80}")
    print(f"5折交叉验证结果汇总")
    print(f"{'='*80}")
    
    # 只显示overall结果的均值
    overall_mean = df[(df['dataset'] == 'overall') & (df['fold'] == 'mean')]
    
    if len(overall_mean) > 0:
        print(f"\n整体准确率 (均值 ± 标准差):")
        print(f"{'模型名称':<15} {'准确率':<12} {'标准差':<10}")
        print("-" * 40)
        
        for _, row in overall_mean.iterrows():
            model_name = row['model_name']
            mean_acc = row['accuracy']
            
            # 找对应的标准差
            std_row = df[(df['model_name'] == model_name) & 
                        (df['dataset'] == 'overall') & 
                        (df['fold'] == 'std')]
            
            if len(std_row) > 0:
                std_acc = std_row.iloc[0]['accuracy']
                print(f"{model_name:<15} {mean_acc:.4f}       ±{std_acc:.4f}")
            else:
                print(f"{model_name:<15} {mean_acc:.4f}       N/A")
    
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "liver_batch_normalize_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/liver/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    # {
    #     'name': 'PCA',
    #     'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/pca/Embeddings_pca.h5ad',
    #     'embedding_key': 'X_pca',
    #     'target_key': 'donor_id',
    #     'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/pca'
    # },
    # {
    #     'name': 'Harmony',
    #     'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/harmony/Embeddings_harmony.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'donor_id',
    #     'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/harmony'
    # },
    # {
    #     'name': 'scVI',
    #     'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scvi/Embeddings_scvi.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'donor_id',
    #     'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/scvi'
    # },
    # {
    #     'name': 'Scanorama',
    #     'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'donor_id',
    #     'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/scanorama'
    # },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'donor_id',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/batch/genecompass'
    },
]


# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH



# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "liver_batch_normalize_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/liver/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    # {
    #     'name': 'PCA',
    #     'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/pca/Embeddings_pca.h5ad',
    #     'embedding_key': 'X_pca',
    #     'target_key': 'cell_type',
    #     'save_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/probing/celltype/pca'
    # },
    # {
    #     'name': 'Harmony',
    #     'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/harmony/Embeddings_harmony.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'cell_type',
    #     'save_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/probing/celltype/harmony'
    # },
    # {
    #     'name': 'scVI',
    #     'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/scvi/Embeddings_scvi.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'cell_type',
    #     'save_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/probing/celltype/scvi'
    # },
    # {
    #     'name': 'Scanorama',
    #     'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/scanorama/Embeddings_scanorama.h5ad',
    #     'embedding_key': 'X_emb',
    #     'target_key': 'cell_type',
    #     'save_path': '/home/wanglinting/LCBERT/Result/benchmark/liver/probing/celltype/scanorama'
    # },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/liver_batch_normalize/probing/celltype/genecompass'
    },
]


# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# # =============================================================================
# # 配置参数和运行
# # =============================================================================

# analysis_name = "Immune_blood_batch_normalize_celltype"

# # CV配置
# CV_CONFIG = {
#     'split_save_dir': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/celltype/data_splits',
#     'n_splits': 5,
#     'max_workers': 7,  # 最大并行数
#     'force_recompute': False  # 是否强制重新计算
# }

# # 结果保存路径
# RESULTS_PATH = '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/cv_results_summary.csv'

# # 模型配置列表
# MODELS_CONFIG = [
#     {
#         'name': 'PCA',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/pca/Embeddings_pca.h5ad',
#         'embedding_key': 'X_pca',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/celltype/pca'
#     },
#     {
#         'name': 'Harmony',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/harmony/Embeddings_harmony.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/celltype/harmony'
#     },
#     {
#         'name': 'scVI',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/scvi/Embeddings_scvi.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/celltype/scvi'
#     },
#     {
#         'name': 'Scanorama',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/scanorama/Embeddings_scanorama.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/celltype/scanorama'
#     },
#     {
#         'name': 'Geneformer',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/geneformer'
#     },
#     {
#         'name': 'scGPT',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/scgpt'
#     },
#     {
#         'name': 'CellPLM',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/cellplm'
#     },
#     {
#         'name': 'UCE',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/uce/Embeddings_uce.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/uce'
#     },
#     {
#         'name': 'scFoundation',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/celltype/scfoundation'
#     },
# ]

# # 执行5折交叉验证线性探测
# print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
# run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
# print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# # 清理内存
# gc.collect()
# # 清理变量
# del CV_CONFIG
# del MODELS_CONFIG
# del RESULTS_PATH


# # =============================================================================
# # 配置参数和运行
# # =============================================================================

# analysis_name = "Immune_blood_batch_normalize_batch"

# # CV配置
# CV_CONFIG = {
#     'split_save_dir': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/batch/data_splits',
#     'n_splits': 5,
#     'max_workers': 7,  # 最大并行数
#     'force_recompute': False  # 是否强制重新计算
# }

# # 结果保存路径
# RESULTS_PATH = '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/cv_results_summary.csv'

# # 模型配置列表
# MODELS_CONFIG = [
#     {
#         'name': 'PCA',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/pca/Embeddings_pca.h5ad',
#         'embedding_key': 'X_pca',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/batch/pca'
#     },
#     {
#         'name': 'Harmony',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/harmony/Embeddings_harmony.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/batch/harmony'
#     },
#     {
#         'name': 'scVI',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/scvi/Embeddings_scvi.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/batch/scvi'
#     },
#     {
#         'name': 'Scanorama',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/scanorama/Embeddings_scanorama.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood/probing/batch/scanorama'
#     },
#     {
#         'name': 'Geneformer',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/geneformer'
#     },
#     {
#         'name': 'scGPT',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/scgpt'
#     },
#     {
#         'name': 'CellPLM',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/cellplm'
#     },
#     {
#         'name': 'UCE',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/uce/Embeddings_uce.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/uce'
#     },
#     {
#         'name': 'scFoundation',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_blood_batch_normalize/probing/batch/scfoundation'
#     },
# ]


# # 执行5折交叉验证线性探测
# print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
# run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
# print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# # 清理内存
# gc.collect()
# # 清理变量
# del CV_CONFIG
# del MODELS_CONFIG
# del RESULTS_PATH



# # =============================================================================
# # 配置参数和运行
# # =============================================================================

# analysis_name = "Immune_lung_batch_normalize_celltype"

# # CV配置
# CV_CONFIG = {
#     'split_save_dir': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/celltype/data_splits',
#     'n_splits': 5,
#     'max_workers': 7,  # 最大并行数
#     'force_recompute': False  # 是否强制重新计算
# }

# # 结果保存路径
# RESULTS_PATH = '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/cv_results_summary.csv'

# # 模型配置列表
# MODELS_CONFIG = [
#     {
#         'name': 'PCA',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/pca/Embeddings_pca.h5ad',
#         'embedding_key': 'X_pca',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/celltype/pca'
#     },
#     {
#         'name': 'Harmony',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/harmony/Embeddings_harmony.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/celltype/harmony'
#     },
#     {
#         'name': 'scVI',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/scvi/Embeddings_scvi.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/celltype/scvi'
#     },
#     {
#         'name': 'Scanorama',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/scanorama/Embeddings_scanorama.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/celltype/scanorama'
#     },
#     {
#         'name': 'Geneformer',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/geneformer'
#     },
#     {
#         'name': 'scGPT',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/scgpt'
#     },
#     {
#         'name': 'CellPLM',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/cellplm'
#     },
#     {
#         'name': 'UCE',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/uce/Embeddings_uce.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/uce'
#     },
#     {
#         'name': 'scFoundation',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'cell_type_level_2',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/celltype/scfoundation'
#     },
# ]


# # 执行5折交叉验证线性探测
# print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
# run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
# print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# # 清理内存
# gc.collect()
# # 清理变量
# del CV_CONFIG
# del MODELS_CONFIG
# del RESULTS_PATH



# # =============================================================================
# # 配置参数和运行
# # =============================================================================

# analysis_name = "Immune_lung_batch_normalize_batch"

# # CV配置
# CV_CONFIG = {
#     'split_save_dir': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/batch/data_splits',
#     'n_splits': 5,
#     'max_workers': 7,  # 最大并行数
#     'force_recompute': False  # 是否强制重新计算
# }

# # 结果保存路径
# RESULTS_PATH = '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/cv_results_summary.csv'

# # 模型配置列表
# MODELS_CONFIG = [
#     {
#         'name': 'PCA',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/pca/Embeddings_pca.h5ad',
#         'embedding_key': 'X_pca',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/batch/pca'
#     },
#     {
#         'name': 'Harmony',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/harmony/Embeddings_harmony.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/batch/harmony'
#     },
#     {
#         'name': 'scVI',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/scvi/Embeddings_scvi.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/batch/scvi'
#     },
#     {
#         'name': 'Scanorama',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/scanorama/Embeddings_scanorama.h5ad',
#         'embedding_key': 'X_emb',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung/probing/batch/scanorama'
#     },
#     {
#         'name': 'Geneformer',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/geneformer'
#     },
#     {
#         'name': 'scGPT',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/scgpt'
#     },
#     {
#         'name': 'CellPLM',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/cellplm'
#     },
#     {
#         'name': 'UCE',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/uce/Embeddings_uce.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/uce'
#     },
#     {
#         'name': 'scFoundation',
#         'adata_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
#         'embedding_key': 'X_emb_batch',
#         'target_key': 'donor_id',
#         'save_path': '/home/wanglinting/LCBERT/Result/benchmark/Immune_lung_batch_normalize/probing/batch/scfoundation'
#     },
# ]


# # 执行5折交叉验证线性探测
# print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
# run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
# print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# # 清理内存
# gc.collect()
# # 清理变量
# del CV_CONFIG
# del MODELS_CONFIG
# del RESULTS_PATH

# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "Immune_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/Immune/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'tissue',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/batch/genecompass'
    },
]


# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "Immune_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/Immune/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type_level_2',
        'save_path': '/home/wanglinting/scFM/Result/Immune_batch_normalize/probing/celltype/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_assay_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_assay/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'assay',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/batch/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_disease_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_disease/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'disease',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/batch/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH



# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_sn_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_sn/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'suspension_type',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/batch/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH



# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_assay_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_assay/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/cv_results_summary.csv'


# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_assay_batch_normalize/probing/celltype/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_disease_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_disease/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_disease_batch_normalize/probing/celltype/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH




# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "HLCA_sn_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/HLCA_sn/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'final_annotation',
        'save_path': '/home/wanglinting/scFM/Result/HLCA_sn_batch_normalize/probing/celltype/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH


# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "limb_batch"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/limb/probing/batch/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'organism',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/batch/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH




# =============================================================================
# 配置参数和运行
# =============================================================================

analysis_name = "limb_celltype"

# CV配置
CV_CONFIG = {
    'split_save_dir': '/home/wanglinting/scFM/Result/limb/probing/celltype/data_splits',
    'n_splits': 5,
    'max_workers': 7,  # 最大并行数
    'force_recompute': False  # 是否强制重新计算
}

# 结果保存路径
RESULTS_PATH = '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/cv_results_summary.csv'

# 模型配置列表
MODELS_CONFIG = [
    {
        'name': 'PCA',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/pca/Embeddings_pca.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/pca'
    },
    {
        'name': 'Harmony',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/harmony/Embeddings_harmony.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/harmony'
    },
    {
        'name': 'scVI',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scvi/Embeddings_scvi.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/scvi'
    },
    {
        'name': 'Scanorama',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scanorama/Embeddings_scanorama.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/scanorama'
    },
    {
        'name': 'Geneformer',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/geneformer/Embeddings_geneformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/geneformer'
    },
    {
        'name': 'scGPT',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scgpt/Embeddings_scgpt.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/scgpt'
    },
    {
        'name': 'CellPLM',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/cellplm/Embeddings_cellplm.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/cellplm'
    },
    {
        'name': 'UCE',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/uce/Embeddings_uce.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/uce'
    },
    {
        'name': 'scFoundation',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/scfoundation/Embeddings_scfoundation.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/scfoundation'
    },
    {
        'name': 'Nicheformer',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/nicheformer/Embeddings_nicheformer.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/nicheformer'
    },
    {
        'name': 'scCello',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/sccello/Embeddings_sccello.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/sccello'
    },
    {
        'name': 'GeneCompass',
        'adata_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/genecompass/Embeddings_genecompass.h5ad',
        'embedding_key': 'X_emb_batch',
        'target_key': 'cell_type',
        'save_path': '/home/wanglinting/scFM/Result/limb_batch_normalize/probing/celltype/genecompass'
    },
]

# 执行5折交叉验证线性探测
print(f"\n{'='*40} {analysis_name} probing分析开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")
run_cv_probing_parallel(cv_config=CV_CONFIG, models_config=MODELS_CONFIG, results_path=RESULTS_PATH)
print(f"\n{'='*40} {analysis_name} probing分析完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*40}")

# 清理内存
gc.collect()
# 清理变量
del CV_CONFIG
del MODELS_CONFIG
del RESULTS_PATH

