#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional
import logging
import multiprocessing

# 设置joblib使用的CPU核心数
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceDatasetWrapper:
    """
    序列数据集包装器,用于数据平衡
    支持过采样、欠采样和SMOTE三种平衡策略
    """
    def __init__(self, dataset: Dataset):
        """
        初始化数据集包装器
        
        Args:
            dataset: PyTorch数据集对象,需要实现__getitem__和__len__方法
        """
        self.dataset = dataset
        self._cache_data()
        
    def _cache_data(self):
        """缓存数据集中的所有样本,用于重采样"""
        try:
            self.sequences = []
            self.labels = []
            self.metadata = []  # 存储其他元数据(video_id, start_idx等)
            
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                self.sequences.append(sample['sequence'].numpy())
                self.labels.append(sample['label'].item())
                
                # 保存除sequence和label之外的所有元数据
                meta = {k: v for k, v in sample.items() if k not in ['sequence', 'label']}
                self.metadata.append(meta)
                
            self.sequences = np.array(self.sequences)
            self.labels = np.array(self.labels)
            
            # 打印原始数据分布
            counter = Counter(self.labels)
            logger.info("原始数据分布:")
            for label, count in counter.items():
                logger.info(f"类别 {int(label)}: {count} 样本")
                
        except Exception as e:
            logger.error(f"数据缓存过程中出错: {str(e)}")
            raise
    
    def balance_dataset(self, 
                       strategy: str = 'oversample',
                       sampling_strategy: Union[str, Dict] = 'auto',
                       random_state: int = 42) -> Dataset:
        """
        对数据集进行平衡处理
        
        Args:
            strategy: 平衡策略,可选['oversample', 'undersample', 'smote']
            sampling_strategy: 采样策略,可以是'auto'或具体的比例字典
            random_state: 随机种子
            
        Returns:
            平衡后的数据集
        """
        try:
            # 重塑序列数据用于重采样
            X = self.sequences.reshape(len(self.sequences), -1)
            y = self.labels
            
            if strategy == 'oversample':
                sampler = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
            elif strategy == 'undersample':
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
            elif strategy == 'smote':
                # 确保邻居数小于最小类别样本数
                min_samples = min(Counter(y).values())
                n_neighbors = min(5, min_samples - 1)
                if n_neighbors < 1:
                    logger.warning("样本数太少,无法使用SMOTE,自动切换为随机过采样")
                    sampler = RandomOverSampler(
                        sampling_strategy=sampling_strategy,
                        random_state=random_state
                    )
                else:
                    sampler = SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=random_state,
                        k_neighbors=n_neighbors,
                    )
            else:
                raise ValueError(f"不支持的平衡策略: {strategy}")
            
            # 执行重采样
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # 获取采样后的索引映射
            if hasattr(sampler, 'sample_indices_'):
                sampled_indices = sampler.sample_indices_
            else:
                # SMOTE没有sample_indices_属性,需要特殊处理
                sampled_indices = list(range(len(self.sequences)))  # 原始样本的索引
                # 为新生成的样本创建特殊的元数据
                num_new_samples = len(y_resampled) - len(y)
                for i in range(num_new_samples):
                    sampled_indices.append(-1)  # 用-1标记新生成的样本
            
            # 重建序列数据的形状
            X_resampled = X_resampled.reshape(-1, *self.sequences.shape[1:])
            
            # 创建平衡后的数据集
            return BalancedDataset(
                sequences=X_resampled,
                labels=y_resampled,
                original_metadata=self.metadata,
                sampled_indices=sampled_indices
            )
            
        except Exception as e:
            logger.error(f"数据平衡过程中出错: {str(e)}")
            raise


class BalancedDataset(Dataset):
    """平衡后的数据集类"""
    def __init__(self, 
                 sequences: np.ndarray,
                 labels: np.ndarray,
                 original_metadata: List[Dict],
                 sampled_indices: List[int]):
        """
        初始化平衡后的数据集
        
        Args:
            sequences: 重采样后的序列数据
            labels: 重采样后的标签
            original_metadata: 原始数据集的元数据
            sampled_indices: 采样使用的索引,用于追踪样本来源
        """
        self.sequences = sequences
        self.labels = labels
        self.original_metadata = original_metadata
        self.sampled_indices = sampled_indices
        
        # 打印平衡后的数据分布
        counter = Counter(self.labels)
        logger.info("\n平衡后的数据分布:")
        for label, count in counter.items():
            logger.info(f"类别 {int(label)}: {count} 样本")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        # 获取元数据
        sampled_idx = self.sampled_indices[idx]
        if sampled_idx >= 0:
            # 使用原始样本的元数据
            metadata = self.original_metadata[sampled_idx]
        else:
            # 为SMOTE生成的新样本创建元数据
            metadata = {
                'video_id': f'synthetic_{idx}',
                'start_idx': 0,
                'end_idx': sequence.shape[0]
            }
        
        return {
            'sequence': sequence,
            'label': label,
            **metadata
        }


def create_balanced_loader(dataset: Dataset,
                         batch_size: int,
                         strategy: str = 'oversample',
                         sampling_strategy: Union[str, Dict] = 'auto',
                         random_state: int = 42,
                         **loader_kwargs) -> torch.utils.data.DataLoader:
    """
    创建平衡的数据加载器的便捷函数
    
    Args:
        dataset: 原始数据集
        batch_size: 批大小
        strategy: 平衡策略
        sampling_strategy: 采样策略
        random_state: 随机种子
        **loader_kwargs: 传递给DataLoader的其他参数
        
    Returns:
        平衡后的数据加载器
    """
    wrapper = SequenceDatasetWrapper(dataset)
    balanced_dataset = wrapper.balance_dataset(
        strategy=strategy,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    return torch.utils.data.DataLoader(
        balanced_dataset,
        batch_size=batch_size,
        **loader_kwargs
    ) 