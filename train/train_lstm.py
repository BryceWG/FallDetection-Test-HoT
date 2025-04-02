#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python train/train_lstm.py --label_file train/frame_data.csv
import os
import sys
import glob
import copy
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from data_balancer import create_balanced_loader

sys.path.append(os.getcwd())

class EarlyStopping:

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

class FallDetectionLSTM(nn.Module):
    """基于LSTM的跌倒检测模型
    输入: 姿态序列数据 [batch_size, sequence_length, feature_dim]
    输出: 二分类结果 [batch_size, 1]
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2, bidirectional=False):
        super(FallDetectionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层 - 支持单向或双向
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional  # 根据参数决定是否使用双向LSTM
        )
        
        # 注意力机制 - 调整维度以适应单向或双向LSTM
        attention_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(attention_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 分类层 - 简化结构并添加BatchNorm，适配单向或双向LSTM输出
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim] 或 [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和得到上下文向量
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim] 或 [batch_size, hidden_dim*2]
        
        # 分类
        output = self.classifier(context)  # [batch_size, 1]
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance
    
    Args:
        alpha (float): 平衡因子，用于处理类别不平衡
        gamma (float): 聚焦参数，用于降低易分类样本的权重
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # 确保数值稳定性
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        # 计算交叉熵
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # 计算focal weight
        p_t = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # 计算最终损失
        loss = weight * ce
        return loss.mean()

class LabelSmoothingBCELoss(nn.Module):
    """带标签平滑的二分类交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # 将目标值从 [0, 1] 平滑到 [smoothing, 1-smoothing]
        smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy(pred, smooth_target)

class PoseSequenceDataset(Dataset):
    """姿态序列数据集
    加载3D或2D姿态数据和对应的标签
    """
    def __init__(self, data_dir, label_file, pose_type='3d', normal_seq_length=30, normal_stride=20,
                 fall_seq_length=30, fall_stride=15, overlap_threshold=0.3,
                 transform=None, test_mode=False, pure_fall=False):
        self.data_dir = data_dir
        self.pose_type = pose_type  # 新增：姿态类型 (2d or 3d)
        self.normal_seq_length = normal_seq_length
        self.normal_stride = normal_stride
        self.fall_seq_length = fall_seq_length
        self.fall_stride = fall_stride
        self.overlap_threshold = overlap_threshold
        self.transform = transform
        self.test_mode = test_mode
        self.pure_fall = pure_fall
        
        # 加载标签数据
        self.labels_df = pd.read_csv(label_file)
        
        # 创建数据索引
        self._create_sequence_samples()
        
        # 初始化标准化参数（将在fit_scaler中设置）
        self.global_mean = None
        self.global_std = None
        self._raw_labels = [sample['label'] for sample in self.samples]  # 缓存标签
        self._feature_dim = self._get_feature_dim() # 获取特征维度
    
    def _get_feature_dim(self):
        """尝试加载一个样本来确定特征维度"""
        if not self.samples:
            # 如果没有样本，无法确定维度，可以给一个默认值或抛出错误
            # 这里假设至少有一个样本可以加载
            print("警告: 数据集中没有样本，无法自动确定特征维度。")
            # 你可能需要根据实际情况返回一个预期的维度或处理此错误
            # 例如，对于3D姿态，可能是 17*3=51，对于2D姿态，可能是 17*2=34
            # 这里暂时返回0，表示需要后续处理
            return 0

        # 尝试加载第一个样本的数据来确定维度
        sample = self.samples[0]
        pose_data = self._load_pose_data(sample['video_id'], sample['label'] == 1)
        if pose_data is None or len(pose_data) == 0:
             print(f"警告: 无法加载样本 {sample['video_id']} 的数据来确定维度。")
             return 0 # 或其他错误处理

        # 假设pose_data的形状是 [frames, keypoints, dims] 或 [frames, features]
        if len(pose_data.shape) == 3: # [frames, keypoints, dims]
            return pose_data.shape[1] * pose_data.shape[2]
        elif len(pose_data.shape) == 2: # 已经是 [frames, features]
            return pose_data.shape[1]
        else:
            print(f"警告: 加载的姿态数据维度不符合预期: {pose_data.shape}")
            return 0

    def fit_scaler(self, indices=None):
        """
        计算指定索引数据的均值和标准差，用于Z-score标准化
        
        Args:
            indices: 用于计算统计量的样本索引列表，如果为None则使用所有样本
        """
        print("\n计算标准化统计信息...")
        all_sequences = []
        
        # 确定要处理的样本
        samples_to_process = [self.samples[i] for i in indices] if indices is not None else self.samples
        
        for sample in tqdm(samples_to_process, desc="加载数据"):
            # 注意：这里调用 _load_pose_data 时可能需要传递 pose_type，但目前 _load_pose_data 内部已使用 self.pose_type
            pose_data = self._load_pose_data(sample['video_id'], sample['label'] == 1)
            if pose_data is None:
                continue
                
            start_idx = sample['start_idx']
            end_idx = sample['end_idx']
            
            sequence = pose_data[start_idx:end_idx]
            sequence = sequence.reshape(len(sequence), -1)
            all_sequences.append(sequence)
            
        if not all_sequences:
            raise ValueError("没有找到有效的序列数据来计算统计量")
            
        all_data = np.concatenate(all_sequences, axis=0)
        self.global_mean = np.mean(all_data, axis=0)
        self.global_std = np.std(all_data, axis=0)
        # 避免除零
        self.global_std[self.global_std < 1e-7] = 1.0
        
        # 验证特征维度是否匹配
        if self.global_mean.shape[0] != self._feature_dim:
             print(f"警告: 计算得到的标准化维度 ({self.global_mean.shape[0]}) 与预期特征维度 ({self._feature_dim}) 不符。")

        print(f"全局均值范围: [{self.global_mean.min():.3f}, {self.global_mean.max():.3f}]")
        print(f"全局标准差范围: [{self.global_std.min():.3f}, {self.global_std.max():.3f}]")
    
    def _load_pose_data(self, video_id, has_fall):
        """加载3D或2D姿态数据
        
        Args:
            video_id: 视频ID
            has_fall: 是否为跌倒视频 (当前未使用，但保留接口)
            
        Returns:
            numpy array: 姿态数据，形状为 [frames, num_keypoints, dims] 或 None
        """
        if self.pose_type == '3d':
            npz_file = os.path.join(self.data_dir, video_id,
                                   'output_3D/output_keypoints_3d.npz')
            key = 'reconstruction'
            expected_dims = 3 # 3D数据预期是 [frames, keypoints, 3]
        elif self.pose_type == '2d':
            npz_file = os.path.join(self.data_dir, video_id,
                                   'input_2D/input_keypoints_2d.npz')
            key = 'reconstruction'  # 2D数据也是使用'reconstruction'作为键名保存的
            expected_dims = 3 # 2D数据预期是 [frames, keypoints, 2]
        else:
            raise ValueError(f"不支持的姿态类型: {self.pose_type}")

        if not os.path.exists(npz_file):
            print(f"警告: 找不到文件 {npz_file}")
            return None

        try:
            data = np.load(npz_file, allow_pickle=True)
            if key not in data:
                 print(f"警告: 在文件 {npz_file} 中找不到键 '{key}'")
                 return None
            pose_data = data[key]
            
            # 检查并修正维度 (特别是针对2D数据可能存在的额外维度)
            if self.pose_type == '2d' and len(pose_data.shape) == 4 and pose_data.shape[0] == 1:
                # 如果是2D数据，且维度是 [1, frames, keypoints, 2]，则移除第一个维度
                pose_data = np.squeeze(pose_data, axis=0)
            elif len(pose_data.shape) != expected_dims:
                print(f"警告: 文件 {npz_file} 加载的数据维度 {pose_data.shape} 与预期 {expected_dims} 不符")
                # 可以选择返回 None 或尝试处理，这里返回 None
                return None
                
            # 确保数据是浮点类型
            return pose_data.astype(np.float32)
        except Exception as e:
            print(f"错误: 加载或处理文件 {npz_file} 时出错: {e}")
            return None
    
    def _create_sequence_samples(self):
        self.samples = []
        
        # 记录数据加载过程的统计信息
        stats = {
            "total_videos": 0,
            "videos_with_missing_data": 0,
            "missing_video_ids": []
        }
        
        # 遍历所有视频
        for _, row in self.labels_df.iterrows():
            stats["total_videos"] += 1
            video_id = row['video_id']
            has_fall = int(row['has_fall'])
            
            # 加载姿态数据
            pose_data = self._load_pose_data(video_id, has_fall)
            if pose_data is None:
                stats["videos_with_missing_data"] += 1
                stats["missing_video_ids"].append(video_id)
                continue
                
            total_frames = len(pose_data)
            
            if has_fall == 0 or not self.pure_fall:
                # 非跌倒视频或非pure_fall模式：使用normal_stride
                if has_fall == 1:
                    # 跌倒视频但非pure_fall模式：使用fall_stride和重叠阈值
                    fall_start = int(row['fall_start_frame']) if not pd.isna(row['fall_start_frame']) else 0
                    fall_end = int(row['fall_end_frame']) if not pd.isna(row['fall_end_frame']) else total_frames
                    stride = self.fall_stride
                    seq_length = self.fall_seq_length
                else:
                    # 非跌倒视频：使用normal_stride
                    stride = self.normal_stride
                    seq_length = self.normal_seq_length
                
                for start_idx in range(0, total_frames - seq_length + 1, stride):
                    end_idx = start_idx + seq_length
                    if end_idx > total_frames:
                        break
                    
                    if has_fall == 1:
                        # 计算当前序列与跌倒片段的重叠长度
                        overlap_start = max(start_idx, fall_start)
                        overlap_end = min(end_idx, fall_end)
                        overlap_length = max(0, overlap_end - overlap_start)
                        
                        # 计算重叠比例
                        overlap_ratio = overlap_length / seq_length
                        
                        # 根据重叠比例判断标签
                        label = 1 if overlap_ratio >= self.overlap_threshold else 0
                        
                        self.samples.append({
                            'video_id': video_id,
                            'label': label,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'overlap_ratio': overlap_ratio
                        })
                    else:
                        self.samples.append({
                            'video_id': video_id,
                            'label': has_fall,
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        })
            else:
                # pure_fall模式下的跌倒视频处理
                fall_start = int(row['fall_start_frame']) if not pd.isna(row['fall_start_frame']) else 0
                fall_end = int(row['fall_end_frame']) if not pd.isna(row['fall_end_frame']) else total_frames
                
                # 1. 添加跌倒前的非跌倒序列
                if fall_start > self.normal_seq_length:
                    for start_idx in range(0, fall_start - self.normal_seq_length + 1, 
                                         self.normal_stride):
                        end_idx = start_idx + self.normal_seq_length
                        if end_idx > fall_start:
                            break
                            
                        self.samples.append({
                            'video_id': video_id,
                            'label': 0,
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        })
                
                # 2. 添加完整的跌倒序列
                fall_length = fall_end - fall_start
                if fall_length >= self.fall_seq_length:
                    # 如果跌倒片段长度足够，使用滑动窗口
                    for start_idx in range(fall_start, fall_end - self.fall_seq_length + 1,
                                         self.fall_stride):
                        end_idx = start_idx + self.fall_seq_length
                        if end_idx > fall_end:
                            break
                            
                        self.samples.append({
                            'video_id': video_id,
                            'label': 1,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'overlap_ratio': 1.0  # 纯跌倒片段
                        })
                else:
                    # 如果跌倒片段不够长，将整个片段作为一个样本
                    self.samples.append({
                        'video_id': video_id,
                        'label': 1,
                        'start_idx': fall_start,
                        'end_idx': fall_end,
                        'overlap_ratio': 1.0
                    })
        
        # 打印数据集统计信息
        total_samples = len(self.samples)
        fall_samples = sum(1 for sample in self.samples if sample['label'] == 1)
        non_fall_samples = total_samples - fall_samples
        
        print(f"\n数据集统计信息:")
        print(f"总视频数: {stats['total_videos']}")
        print(f"成功加载视频数: {stats['total_videos'] - stats['videos_with_missing_data']}")
        print(f"未找到{self.pose_type}姿态数据的视频数: {stats['videos_with_missing_data']}")
        if stats["videos_with_missing_data"] > 0:
            print(f"前5个缺失数据的视频ID: {stats['missing_video_ids'][:5]}")
            
            # 尝试检查第一个缺失视频的文件路径是否存在
            if stats["missing_video_ids"]:
                first_missing = stats["missing_video_ids"][0]
                if self.pose_type == '3d':
                    expected_path = os.path.join(self.data_dir, first_missing, 'output_3D/output_keypoints_3d.npz')
                else:
                    expected_path = os.path.join(self.data_dir, first_missing, 'input_2D/input_keypoints_2d.npz')
                print(f"检查路径是否存在: {expected_path}")
                print(f"路径存在: {os.path.exists(expected_path)}")
                
                # 检查父目录
                parent_dir = os.path.dirname(expected_path)
                print(f"父目录存在: {os.path.exists(parent_dir)}")
                
                # 如果父目录存在，列出其内容
                if os.path.exists(parent_dir):
                    print(f"父目录 {parent_dir} 内容:")
                    try:
                        files = os.listdir(parent_dir)
                        for file in files[:10]:  # 只显示前10个文件
                            print(f"  - {file}")
                        if len(files) > 10:
                            print(f"  ... 以及 {len(files) - 10} 个其他文件")
                    except Exception as e:
                        print(f"无法列出目录内容: {e}")
        
        print(f"总样本数: {total_samples}")
        print(f"跌倒样本数: {fall_samples}")
        print(f"非跌倒样本数: {non_fall_samples}")
        if total_samples > 0:
            print(f"跌倒/非跌倒比例: {fall_samples/non_fall_samples:.2f}")
            
        # 如果没有样本，给出更明确的警告
        if total_samples == 0:
            print(f"\n警告：未找到任何有效的{self.pose_type}姿态数据样本！")
            print(f"请检查:")
            print(f"1. 数据目录 {self.data_dir} 是否包含视频子目录")
            print(f"2. 视频子目录中是否包含 {'input_2D/input_keypoints_2d.npz' if self.pose_type == '2d' else 'output_3D/output_keypoints_3d.npz'} 文件")
            print(f"3. 标签文件中的视频ID是否与数据目录中的子目录名称匹配")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载姿态数据 (使用self.pose_type)
        pose_data = self._load_pose_data(sample['video_id'], sample['label'] == 1)
        
        if pose_data is None:
             # 处理数据加载失败的情况，例如返回一个空字典或抛出异常
             # 这里我们返回一个包含标识符的字典，让DataLoader的collate_fn处理或跳过
             print(f"警告: 无法加载样本 {idx} 的数据 (video_id: {sample['video_id']})")
             # 返回一个可以被 collate_fn 识别并过滤掉的特殊值，或者你需要修改 collate_fn
             # 或者，更简单的方式是在 _create_sequence_samples 中就跳过无法加载的视频
             # 为保持简单，我们先假设 _create_sequence_samples 已过滤掉无法加载的视频
             # 如果仍然出现 None，这里应该抛出更明确的错误
             raise RuntimeError(f"无法加载样本 {idx} (video_id: {sample['video_id']}) 的姿态数据")

        # 截取指定范围的序列
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # 确保序列长度一致
        sequence = pose_data[start_idx:end_idx]
        
        # 如果序列长度不足，使用重复填充
        if len(sequence) < self.normal_seq_length:
            # 先展平每一帧的关键点数据
            flattened_sequence = sequence.reshape(len(sequence), -1)
            # 计算需要填充的长度
            padding_length = self.normal_seq_length - len(sequence)
            # 复制最后一帧进行填充
            last_frame = flattened_sequence[-1:]
            padding = np.tile(last_frame, (padding_length, 1))
            # 拼接原序列和填充序列
            flattened_sequence = np.concatenate([flattened_sequence, padding], axis=0)
        else:
            # 如果序列长度足够，直接截取指定长度并展平
            sequence = sequence[:self.normal_seq_length]
            flattened_sequence = sequence.reshape(self.normal_seq_length, -1)
        
        # 验证展平后的特征维度是否正确
        if flattened_sequence.shape[1] != self._feature_dim:
            print(f"警告: 样本 {idx} (video_id: {sample['video_id']}) 的展平维度 ({flattened_sequence.shape[1]}) 与数据集预期维度 ({self._feature_dim}) 不符。")
            # 你可能需要根据情况处理这个不匹配，例如跳过这个样本或重新调整
            # 这里暂时继续，但标准化可能会出错
        
        # 使用Z-score标准化
        if self.transform:
            flattened_sequence = self.transform(flattened_sequence)
        elif self.global_mean is not None and self.global_std is not None:
            flattened_sequence = (flattened_sequence - self.global_mean) / self.global_std
        else:
            raise RuntimeError("在使用数据集之前必须调用fit_scaler来计算标准化参数")
        
        # 转换为tensor
        sequence_tensor = torch.FloatTensor(flattened_sequence)
        label_tensor = torch.FloatTensor([sample['label']])
        
        return {
            'sequence': sequence_tensor,
            'label': label_tensor,
            'video_id': sample['video_id'],
            'start_idx': start_idx,
            'end_idx': end_idx
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, save_dir='./checkpoints/fall_detection'):
    """
    训练模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * sequences.size(0)
            pred = (outputs >= 0.5).float()
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
            
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * sequences.size(0)
                pred = (outputs >= 0.5).float()
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调整
        if scheduler:
            scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"保存最佳模型, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 每个epoch结束后显示指标
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # 如果触发早停，恢复最佳模型并结束训练
        if early_stopping.early_stop:
            print("早停触发，恢复最佳模型")
            model.load_state_dict(early_stopping.best_model)
            break
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘制训练曲线
    plot_training_history(history, save_dir)
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot training history curves
    """
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Training Accuracy')
    plt.plot(history['val_accs'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, test_loader, criterion, device, save_dir='./checkpoints/fall_detection'):
    """
    Evaluate model
    """
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    test_videos = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            video_ids = batch['video_id']
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * sequences.size(0)
            pred = (outputs >= 0.5).float()
            
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_videos.extend(video_ids)
    
    # Calculate average loss
    test_loss = test_loss / len(test_loader.dataset)
    
    # Binarize predictions
    binary_preds = (np.array(test_preds) >= 0.5).astype(int)
    true_labels = np.array(test_labels).astype(int)
    
    # Calculate performance metrics
    classification_rep = classification_report(true_labels, binary_preds, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, binary_preds)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'predictions': test_preds,
        'binary_predictions': binary_preds,
        'true_labels': true_labels,
        'video_ids': test_videos,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix
    }
    
    with open(os.path.join(save_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print evaluation results
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, binary_preds))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    class_names = ['Normal', 'Fall']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # Show numbers in cells
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return results

def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='跌倒检测模型训练')
    parser.add_argument('--data_dir', type=str, default='./demo/output/', help='姿态数据根目录')
    parser.add_argument('--label_file', type=str, required=False, help='标签文件路径(CSV格式)')
    parser.add_argument('--pose_type', type=str, default='3d', choices=['2d', '3d'],
                       help='使用的姿态数据类型 (2d or 3d)')
    
    # 非跌倒视频的序列参数
    parser.add_argument('--normal_seq_length', type=int, default=30, help='非跌倒视频序列长度')
    parser.add_argument('--normal_stride', type=int, default=30, help='非跌倒视频滑动步长')
    
    # 跌倒视频的序列参数
    parser.add_argument('--fall_seq_length', type=int, default=30, help='跌倒视频序列长度')
    parser.add_argument('--fall_stride', type=int, default=10, help='跌倒视频滑动步长')
    
    # 数据集划分参数
    parser.add_argument('--test_ratio', type=float, default=0.25, help='测试集占总数据的比例')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='验证集占训练数据的比例')
    
    # 交叉验证参数 - 简化参数设置
    parser.add_argument('--n_splits', type=int, default=0, 
                       help='交叉验证折数,大于0则启用交叉验证,默认使用分层K折交叉验证')
    
    # 数据平衡参数
    parser.add_argument('--balance_strategy', type=str, default='smote', 
                       choices=['none', 'oversample', 'undersample', 'smote'],
                       help='数据平衡策略')
    parser.add_argument('--target_ratio', type=float, default=1.0,
                       help='目标类别比例(跌倒:非跌倒),仅在balance_strategy不为none时有效')
    
    # 模型参数
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2正则化系数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    parser.add_argument('--lstm_bi', action='store_true', help='是否使用双向LSTM架构')
    
    # --save_dir 定义时不设置复杂的默认值
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='模型保存目录 (默认: ./checkpoint/fall_detection_lstm_{pose_type}/{timestamp})')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载checkpoint路径')
    parser.add_argument('--overlap_threshold', type=float, default=0.3,
                       help='跌倒判定的重叠比例阈值(0-1之间),仅对跌倒视频有效')
    parser.add_argument('--pure_fall', action='store_true',
                       help='启用纯跌倒模式,从标签文件中读取跌倒片段起始帧')
    
    # 只在这里进行一次完整的参数解析
    args = parser.parse_args()
    
    # 如果用户没有提供 save_dir，则在解析后生成默认值
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        args.save_dir = f'./checkpoint/fall_detection_lstm_{args.pose_type}/{timestamp}'
        
    return args

def save_training_summary(args, train_results, test_results, save_dir, dataset):

    # 提取需要保存的参数
    params = {
        "data_params": {
            "data_dir": args.data_dir,
            "pose_type": args.pose_type,
            "normal_seq_length": args.normal_seq_length,
            "normal_stride": args.normal_stride,
            "fall_seq_length": args.fall_seq_length,
            "fall_stride": args.fall_stride,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio
        },
        "model_params": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout
        },
        "train_params": {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed
        },
        "normalization_params": {
            "mean": dataset.global_mean.tolist() if dataset.global_mean is not None else None,
            "std": dataset.global_std.tolist() if dataset.global_std is not None else None
        }
    }
    
    # 提取训练结果
    train_summary = {
        "best_val_loss": float(train_results["val_loss"]),
        "best_val_accuracy": float(train_results["val_acc"]),
        "best_epoch": train_results["epoch"]
    }
    
    # 提取测试结果
    test_summary = {
        "test_loss": float(test_results["test_loss"]),
        "classification_report": {
            k: v for k, v in test_results["classification_report"].items()
            if k in ["0", "1", "accuracy", "macro avg", "weighted avg"]
        }
    }
    
    # 合并所有信息
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params,
        "training_results": train_summary,
        "test_results": test_summary
    }
    
    # 保存为JSON文件
    summary_file = os.path.join(save_dir, "training_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n训练摘要已保存至: {summary_file}")
    return summary

def run_cross_validation(full_dataset, args, device):
    """
    执行K折交叉验证
    """
    print(f"\n开始{args.n_splits}折分层交叉验证...")
    
    # 使用分层K折交叉验证
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = kfold.split(np.arange(len(full_dataset)), full_dataset._raw_labels)
    
    # 存储每折的指标
    fold_metrics = []
    
    for fold, (train_val_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*20} 第 {fold+1}/{args.n_splits} 折 {'='*20}")
        
        # 为当前折创建保存目录
        fold_save_dir = os.path.join(args.save_dir, f'fold_{fold+1}')
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # 进一步划分训练集和验证集
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=args.val_ratio,
            stratify=[full_dataset._raw_labels[i] for i in train_val_idx],
            random_state=args.seed
        )
        
        # 使用训练集数据计算标准化参数
        full_dataset.fit_scaler(train_idx)
        
        # 创建数据子集
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        if args.balance_strategy != 'none':
            # 使用数据平衡策略
            train_labels = [full_dataset._raw_labels[i] for i in train_idx]
            label_counts = Counter(train_labels)
            
            if args.target_ratio != 1.0:
                max_count = max(label_counts.values())
                target_counts = {
                    0: int(max_count),
                    1: int(max_count * args.target_ratio)
                }
            else:
                target_counts = 'auto'
                
            print(f"\n使用{args.balance_strategy}策略平衡数据...")
            train_loader = create_balanced_loader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                strategy=args.balance_strategy,
                sampling_strategy=target_counts,
                shuffle=True,
                num_workers=4
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4
            )
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        input_dim = full_dataset._feature_dim
        model = FallDetectionLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.lstm_bi
        ).to(device)
        
        # 定义损失函数和优化器
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        
        # 训练模型
        print(f"\n开始第 {fold+1} 折训练...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            device=device,
            scheduler=scheduler,
            save_dir=fold_save_dir
        )
        
        # 加载最佳模型进行测试
        best_model_path = os.path.join(fold_save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_results = {
                "val_loss": checkpoint['val_loss'],
                "val_acc": checkpoint['val_acc'],
                "epoch": checkpoint['epoch']
            }
        
        # 评估模型
        print(f"评估第 {fold+1} 折模型...")
        test_results = evaluate_model(model, test_loader, criterion, device, fold_save_dir)
        
        # 记录当前折的指标
        fold_metrics.append({
            'fold': fold + 1,
            'train_results': best_val_results,
            'test_results': test_results
        })
    
    # 计算并保存交叉验证平均指标
    avg_metrics = calculate_cv_metrics(fold_metrics, args.save_dir)
    
    return fold_metrics, avg_metrics

def calculate_cv_metrics(fold_metrics, save_dir):
    """
    计算交叉验证的平均指标
    """
    # 提取各折的测试指标
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    for fold_data in fold_metrics:
        test_results = fold_data['test_results']
        test_losses.append(test_results['test_loss'])
        
        # 从分类报告中提取指标
        class_report = test_results['classification_report']
        test_accuracies.append(class_report['accuracy'])
        
        # 提取跌倒类(类别1)的指标
        test_precisions.append(class_report['1']['precision'])
        test_recalls.append(class_report['1']['recall'])
        test_f1_scores.append(class_report['1']['f1-score'])
    
    # 计算平均指标
    avg_metrics = {
        'avg_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'avg_accuracy': np.mean(test_accuracies),
        'std_accuracy': np.std(test_accuracies),
        'avg_precision': np.mean(test_precisions),
        'std_precision': np.std(test_precisions),
        'avg_recall': np.mean(test_recalls),
        'std_recall': np.std(test_recalls),
        'avg_f1': np.mean(test_f1_scores),
        'std_f1': np.std(test_f1_scores)
    }
    
    # 打印平均指标
    print("\n交叉验证平均指标:")
    print(f"测试损失: {avg_metrics['avg_test_loss']:.4f} ± {avg_metrics['std_test_loss']:.4f}")
    print(f"准确率: {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"精确率: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"召回率: {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    print(f"F1分数: {avg_metrics['avg_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    
    # 保存交叉验证结果
    cv_results = {
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics
    }
    
    with open(os.path.join(save_dir, 'cv_results.json'), 'w', encoding='utf-8') as f:
        # 将numpy数组转换为列表以便JSON序列化
        json_results = json.dumps(cv_results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=4, ensure_ascii=False)
        f.write(json_results)
    
    # 绘制交叉验证结果图表
    plot_cv_metrics(fold_metrics, avg_metrics, save_dir)
    
    return avg_metrics

def plot_cv_metrics(fold_metrics, avg_metrics, save_dir):
    """
    绘制交叉验证指标图表
    """
    # 提取各折的测试准确率、精确率、召回率和F1分数
    folds = [data['fold'] for data in fold_metrics]
    accuracies = [data['test_results']['classification_report']['accuracy'] for data in fold_metrics]
    precisions = [data['test_results']['classification_report']['1']['precision'] for data in fold_metrics]
    recalls = [data['test_results']['classification_report']['1']['recall'] for data in fold_metrics]
    f1_scores = [data['test_results']['classification_report']['1']['f1-score'] for data in fold_metrics]
    
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    
    plt.plot(folds, accuracies, 'o-', label='Accuracy')
    plt.plot(folds, precisions, 's-', label='Precision')
    plt.plot(folds, recalls, '^-', label='Recall')
    plt.plot(folds, f1_scores, 'd-', label='F1 Score')
    
    # 添加平均值水平线
    plt.axhline(y=avg_metrics['avg_accuracy'], color='blue', linestyle='--', alpha=0.5, label='Mean Accuracy')
    plt.axhline(y=avg_metrics['avg_precision'], color='orange', linestyle='--', alpha=0.5, label='Mean Precision')
    plt.axhline(y=avg_metrics['avg_recall'], color='green', linestyle='--', alpha=0.5, label='Mean Recall')
    plt.axhline(y=avg_metrics['avg_f1'], color='red', linestyle='--', alpha=0.5, label='Mean F1')
    
    plt.xlabel('Fold Number')
    plt.ylabel('Metric Value')
    plt.title('Cross Validation Metrics by Fold')
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_metrics.png'))
    plt.close()

def main():

    args = process_args()
    # 在解析参数后，重新构建默认的 save_dir，因为默认值是在解析前基于初步的 args 创建的
    if args.save_dir == f'./checkpoint/fall_detection_lstm_{args.pose_type}/{datetime.now().strftime("%Y-%m-%d-%H%M")}': # 检查是否是未替换 pose_type 的默认路径
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
        args.save_dir = f'./checkpoint/fall_detection_lstm_{args.pose_type}/{timestamp}'

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    if os.path.exists(args.label_file):
        print(f"加载标签文件: {args.label_file}")
    else:
        print(f"错误: 标签文件不存在 - {args.label_file}")
        return
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在 - {args.data_dir}")
        return
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据集和加载器
    print("创建数据集...")
    full_dataset = PoseSequenceDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        pose_type=args.pose_type,
        normal_seq_length=args.normal_seq_length,
        normal_stride=args.normal_stride,
        fall_seq_length=args.fall_seq_length,
        fall_stride=args.fall_stride,
        overlap_threshold=args.overlap_threshold,
        pure_fall=args.pure_fall
    )
    
    print(f"数据集大小: {len(full_dataset)}")
    
    # 检查是否执行交叉验证
    if args.n_splits > 0:
        fold_metrics, avg_metrics = run_cross_validation(full_dataset, args, device)
        print("\n交叉验证完成!")
        return avg_metrics
    
    # 标准训练流程 (非交叉验证模式)
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=args.test_ratio,
        stratify=full_dataset._raw_labels,  # 使用缓存的标签
        random_state=args.seed
    )
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=args.val_ratio,
        stratify=[full_dataset._raw_labels[i] for i in train_indices],  # 使用缓存的标签
        random_state=args.seed
    )
    
    # 使用训练集数据计算标准化参数
    full_dataset.fit_scaler(train_indices)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    if args.balance_strategy != 'none':
        # 使用数据平衡策略
        # 计算目标样本数
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        label_counts = Counter(train_labels)
        
        if args.target_ratio != 1.0:
            # 使用自定义比例
            max_count = max(label_counts.values())
            target_counts = {
                0: int(max_count),  # 非跌倒类保持不变
                1: int(max_count * args.target_ratio)  # 跌倒类根据比例调整
            }
        else:
            # 自动平衡到相同数量
            target_counts = 'auto'
            
        print(f"\n使用{args.balance_strategy}策略平衡数据...")
        train_loader = create_balanced_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            strategy=args.balance_strategy,
            sampling_strategy=target_counts,
            shuffle=True,
            num_workers=4
        )
    else:
        # 使用原始数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    # 验证集和测试集不需要平衡
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 计算输入特征维度
    # 从数据集中获取特征维度
    input_dim = full_dataset._feature_dim
    if input_dim == 0:
         print("错误: 无法确定输入特征维度，请检查数据加载过程。")
         return # 或者设置一个默认值并打印警告

    print(f"输入特征维度 (基于 {args.pose_type} 数据): {input_dim}")
    
    # 创建模型
    model = FallDetectionLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.lstm_bi
    ).to(device)
    
    print(model)
    
    # 定义损失函数和优化器
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 加载checkpoint(如果有)
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"加载checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"从epoch {start_epoch}继续训练")
        else:
            print(f"警告: checkpoint文件不存在 - {args.checkpoint}")
    
    # 训练模型
    print("\n开始训练...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        scheduler=scheduler,
        save_dir=args.save_dir
    )
    print("训练完成!")
    
    # 加载最佳模型进行测试
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_results = {
            "val_loss": checkpoint['val_loss'],
            "val_acc": checkpoint['val_acc'],
            "epoch": checkpoint['epoch']
        }
    
    # 评估模型
    print("开始模型评估...")
    test_results = evaluate_model(model, test_loader, criterion, device, args.save_dir)
    
    # 保存训练摘要
    summary = save_training_summary(
        args=args,
        train_results=best_val_results,
        test_results=test_results,
        save_dir=args.save_dir,
        dataset=full_dataset
    )
    print("评估完成!")
    
    return model, history, test_results, summary
    
if __name__ == "__main__":
    main()