#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python train/train_stgcn.py --label_file train/frame_data.csv --data_dir ./demo/output/ --save_dir ./checkpoint/fall_detection_stgcn/
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.getcwd())


# 定义人体骨架的连接关系
# 17个关键点的连接关系，用于构建图卷积的邻接矩阵
# COCO关键点顺序: [鼻子，左眼，右眼，左耳，右耳，左肩，右肩，左肘，右肘，左腕，右腕，左髋，右髋，左膝，右膝，左踝，右踝]
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # 鼻子-眼睛
    (1, 3), (2, 4),  # 眼睛-耳朵
    (5, 7), (7, 9),  # 左肩-左肘-左腕
    (6, 8), (8, 10), # 右肩-右肘-右腕
    (5, 6),          # 左肩-右肩
    (5, 11), (6, 12), # 肩-髋
    (11, 13), (13, 15), # 左髋-左膝-左踝
    (12, 14), (14, 16), # 右髋-右膝-右踝
    (11, 12)         # 左髋-右髋
]


class GraphConvolution(nn.Module):
    """
    简单的图卷积层
    """
    def __init__(self, in_channels, out_channels, adj):
        super(GraphConvolution, self).__init__()
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # x: [batch_size, num_nodes, in_channels]
        batch_size, num_nodes, in_channels = x.size()
        
        # 图卷积: X' = AXW
        support = torch.matmul(x, self.weight)  # XW: [batch_size, num_nodes, out_channels]
        output = torch.matmul(self.adj.to(x.device), support)  # AXW: [batch_size, num_nodes, out_channels]
        output = output + self.bias
        
        return output


class STGCNBlock(nn.Module):
    """
    时空图卷积块
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, adj, dropout=0.2):
        super(STGCNBlock, self).__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, adj)
        
        # 时间卷积
        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_channels * adj.size(0)),
            nn.ReLU(),
            nn.Conv1d(
                out_channels * adj.size(0),
                out_channels * adj.size(0),
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                groups=adj.size(0)
            ),
            nn.BatchNorm1d(out_channels * adj.size(0)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels * adj.size(0),
                    out_channels * adj.size(0),
                    kernel_size=1,
                    stride=stride,
                    groups=adj.size(0)
                ),
                nn.BatchNorm1d(out_channels * adj.size(0))
            )
        else:
            self.residual = lambda x: x
        
    def forward(self, x):
        # x: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.size()
        
        # 准备残差连接
        # 将输入重塑为 [batch_size, num_nodes*in_channels, seq_len]
        res = x.reshape(batch_size, seq_len, num_nodes * in_channels).permute(0, 2, 1)
        res = self.residual(res)
        
        # 应用GCN到每个时间步
        x_gcn = []
        for t in range(seq_len):
            # [batch_size, num_nodes, in_channels]
            gcn_out = self.gcn(x[:, t])  # [batch_size, num_nodes, out_channels]
            x_gcn.append(gcn_out)
        
        # 堆叠GCN输出
        x_gcn = torch.stack(x_gcn, dim=1)  # [batch_size, seq_len, num_nodes, out_channels]
        
        # 重塑以应用TCN
        out_channels = x_gcn.size(-1)
        x_tcn = x_gcn.reshape(batch_size, seq_len, num_nodes * out_channels).permute(0, 2, 1)
        x_tcn = self.tcn(x_tcn)
        
        # 残差连接
        out = x_tcn + res
        
        # 重塑回原始格式
        out = out.permute(0, 2, 1).reshape(batch_size, -1, num_nodes, out_channels)
        
        return out


class FallDetectionSTGCN(nn.Module):
    """
    基于ST-GCN的跌倒检测模型
    输入: 姿态序列数据 [batch_size, sequence_length, num_joints * 3]
    输出: 二分类结果 [batch_size, 1]
    """
    def __init__(self, num_joints=17, in_channels=3, hidden_dim=64, num_layers=3, dropout=0.2):
        super(FallDetectionSTGCN, self).__init__()
        
        # 创建邻接矩阵
        self.adj = self._build_adjacency_matrix(num_joints)
        
        # 特征转换
        self.input_transform = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        
        # ST-GCN层
        self.st_gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * 2
            out_channels = hidden_dim * 2
            self.st_gcn_layers.append(
                STGCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    adj=self.adj,
                    dropout=dropout
                )
            )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * num_joints, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        
    def _build_adjacency_matrix(self, num_joints):
        """构建邻接矩阵"""
        # 初始化邻接矩阵
        adj = torch.zeros((num_joints, num_joints))
        
        # 添加自连接
        for i in range(num_joints):
            adj[i, i] = 1
        
        # 添加关键点之间的连接
        for i, j in SKELETON_CONNECTIONS:
            if i < num_joints and j < num_joints:
                adj[i, j] = 1
                adj[j, i] = 1  # 无向图
        
        # 归一化邻接矩阵
        D = torch.sum(adj, dim=1)
        D_sqrt_inv = torch.diag(torch.pow(D, -0.5))
        adj_normalized = torch.mm(torch.mm(D_sqrt_inv, adj), D_sqrt_inv)
        
        return adj_normalized
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, num_joints * 3]
        batch_size, seq_len, features = x.size()
        
        # 确保输入维度正确
        if features % self.num_joints != 0:
            raise ValueError(f"输入特征维度 {features} 不能被关节数量 {self.num_joints} 整除")
        
        # 计算每个关节的通道数
        channels_per_joint = features // self.num_joints
        
        # 重塑为 [batch_size, seq_len, num_joints, channels_per_joint]
        x = x.reshape(batch_size, seq_len, self.num_joints, channels_per_joint)
        
        # 特征变换: 将3D坐标转换为hidden_dim维特征
        # 首先将x变形为适合Conv1d的形状
        x_in = x.permute(0, 3, 1, 2)  # [batch_size, channels_per_joint, seq_len, num_joints]
        x_in = x_in.reshape(batch_size, channels_per_joint, -1)  # [batch_size, channels_per_joint, seq_len*num_joints]
        x_transformed = self.input_transform(x_in)  # [batch_size, hidden_dim, seq_len*num_joints]
        x_transformed = x_transformed.reshape(batch_size, self.hidden_dim, seq_len, self.num_joints)
        x_transformed = x_transformed.permute(0, 2, 3, 1)  # [batch_size, seq_len, num_joints, hidden_dim]
        
        # 应用ST-GCN层
        x = x_transformed
        for st_gcn in self.st_gcn_layers:
            x = st_gcn(x)
        
        # 全局池化 - 时间维度
        x = torch.mean(x, dim=1)  # [batch_size, num_joints, hidden_dim*2]
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 分类预测
        x = self.classifier(x)
        
        return x


class PoseSequenceDataset(Dataset):
    """
    姿态序列数据集
    加载3D姿态数据并转换为适合ST-GCN的格式
    """
    def __init__(self, data_dir, label_file, normal_seq_length=30, normal_stride=20,
                 fall_seq_length=30, fall_stride=15, overlap_threshold=0.3,
                 transform=None, test_mode=False):
        self.data_dir = data_dir
        self.normal_seq_length = normal_seq_length
        self.normal_stride = normal_stride
        self.fall_seq_length = fall_seq_length
        self.fall_stride = fall_stride
        self.overlap_threshold = overlap_threshold
        self.transform = transform
        self.test_mode = test_mode
        
        # 加载标签数据
        self.labels_df = pd.read_csv(label_file)
        
        # 创建数据索引
        self._create_sequence_samples()
        
    def _create_sequence_samples(self):
        """
        创建序列样本索引
        - 非跌倒视频: 使用normal_seq_length和normal_stride
        - 跌倒视频: 使用fall_seq_length和fall_stride,根据overlap_threshold判断标签
        """
        self.samples = []
        
        # 遍历所有视频
        for _, row in self.labels_df.iterrows():
            video_id = row['video_id']
            npz_file = os.path.join(self.data_dir, video_id, 'output_3D/output_keypoints_3d.npz')
            
            if not os.path.exists(npz_file):
                print(f"警告: 找不到文件 {npz_file}")
                continue
                
            # 加载3D姿态数据
            pose_data = np.load(npz_file, allow_pickle=True)['reconstruction']
            total_frames = len(pose_data)
            
            has_fall = int(row['has_fall'])
            
            if has_fall == 0:
                # 非跌倒视频: 使用normal_stride
                for start_idx in range(0, total_frames - self.normal_seq_length + 1, self.normal_stride):
                    end_idx = start_idx + self.normal_seq_length
                    
                    # 如果剩余帧数不足一个完整序列,则跳过
                    if end_idx > total_frames:
                        break
                        
                    self.samples.append({
                        'video_id': video_id,
                        'pose_file': npz_file,
                        'label': 0,  # 非跌倒
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
            else:
                # 跌倒视频: 使用fall_stride
                fall_start = int(row['fall_start_frame'])
                fall_end = int(row['fall_end_frame'])
                fall_duration = fall_end - fall_start + 1
                
                for start_idx in range(0, total_frames - self.fall_seq_length + 1, self.fall_stride):
                    end_idx = start_idx + self.fall_seq_length
                    
                    # 如果剩余帧数不足一个完整序列,则跳过
                    if end_idx > total_frames:
                        break
                    
                    # 计算当前序列与跌倒帧段的重叠部分
                    overlap_start = max(start_idx, fall_start)
                    overlap_end = min(end_idx, fall_end)
                    
                    if overlap_end > overlap_start:
                        # 存在重叠,计算重叠比例
                        overlap_length = overlap_end - overlap_start + 1
                        overlap_ratio = overlap_length / self.fall_seq_length
                        
                        # 根据overlap_threshold判断是否为跌倒
                        is_fall = overlap_ratio >= self.overlap_threshold
                    else:
                        # 无重叠
                        is_fall = False
                    
                    self.samples.append({
                        'video_id': video_id,
                        'pose_file': npz_file,
                        'label': int(is_fall),
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'overlap_ratio': overlap_ratio if overlap_end > overlap_start else 0
                    })
        
        # 打印数据集统计信息
        total_samples = len(self.samples)
        fall_samples = sum(1 for sample in self.samples if sample['label'] == 1)
        non_fall_samples = total_samples - fall_samples
        
        print(f"\n数据集统计信息:")
        print(f"总样本数: {total_samples}")
        print(f"跌倒样本数: {fall_samples}")
        print(f"非跌倒样本数: {non_fall_samples}")
        print(f"跌倒/非跌倒比例: {fall_samples/non_fall_samples:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载3D姿态数据
        pose_data = np.load(sample['pose_file'], allow_pickle=True)['reconstruction']
        
        # 截取指定范围的序列
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # 确保序列长度一致
        if end_idx - start_idx < self.normal_seq_length:
            # 如果序列不够长,进行填充
            padding_length = self.normal_seq_length - (end_idx - start_idx)
            sequence = pose_data[start_idx:end_idx]
            
            # 通过重复最后一帧进行填充
            last_frame = sequence[-1:].repeat(padding_length, axis=0)
            sequence = np.concatenate([sequence, last_frame], axis=0)
        else:
            # 如果序列足够长,直接截取指定长度
            sequence = pose_data[start_idx:start_idx + self.normal_seq_length]
        
        # 对于ST-GCN，我们需要保持关节的空间结构
        # 原始形状: [seq_len, num_joints, 3]
        # 转换为: [seq_len, num_joints * 3] 用于模型输入
        flattened_sequence = sequence.reshape(self.normal_seq_length, -1)
        
        # 特征标准化
        if self.transform:
            flattened_sequence = self.transform(flattened_sequence)
        else:
            # 简单的min-max标准化
            seq_min = flattened_sequence.min(axis=0, keepdims=True)
            seq_max = flattened_sequence.max(axis=0, keepdims=True)
            seq_range = seq_max - seq_min
            # 避免除零
            seq_range[seq_range == 0] = 1.0
            flattened_sequence = (flattened_sequence - seq_min) / seq_range
        
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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, save_dir='./checkpoints/fall_detection_stgcn'):
    """
    训练模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
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
    绘制训练历史曲线
    """
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失')
    plt.plot(history['val_losses'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='训练准确率')
    plt.plot(history['val_accs'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def evaluate_model(model, test_loader, criterion, device, save_dir='./checkpoints/fall_detection_stgcn'):
    """
    评估模型
    """
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    test_videos = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            video_ids = batch['video_id']
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 统计
            test_loss += loss.item() * sequences.size(0)
            pred = (outputs >= 0.5).float()
            
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_videos.extend(video_ids)
    
    # 计算平均损失
    test_loss = test_loss / len(test_loader.dataset)
    
    # 二值化预测
    binary_preds = (np.array(test_preds) >= 0.5).astype(int)
    true_labels = np.array(test_labels).astype(int)
    
    # 计算性能指标
    classification_rep = classification_report(true_labels, binary_preds, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, binary_preds)
    
    # 保存结果
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
    
    # 打印评估结果
    print(f"测试损失: {test_loss:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, binary_preds))
    print("\n混淆矩阵:")
    print(conf_matrix)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    class_names = ['正常', '跌倒']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # 在单元格中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return results


def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='ST-GCN跌倒检测模型训练')
    parser.add_argument('--data_dir', type=str, default='./demo/output/', help='3D姿态数据目录')
    parser.add_argument('--label_file', type=str, required=False, help='标签文件路径(CSV格式)')
    
    # 非跌倒视频的序列参数
    parser.add_argument('--normal_seq_length', type=int, default=30, help='非跌倒视频序列长度')
    parser.add_argument('--normal_stride', type=int, default=40, help='非跌倒视频滑动步长')
    
    # 跌倒视频的序列参数
    parser.add_argument('--fall_seq_length', type=int, default=30, help='跌倒视频序列长度')
    parser.add_argument('--fall_stride', type=int, default=10, help='跌倒视频滑动步长')
    parser.add_argument('--overlap_threshold', type=float, default=0.5, help='跌倒判定的重叠比例阈值')
    
    # 数据集划分参数
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集占总数据的比例')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='验证集占训练数据的比例')
    
    # ST-GCN特有参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='GCN隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='ST-GCN层数')
    
    # 其他训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 生成带时间戳的保存目录
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
    default_save_dir = f'./checkpoint/fall_detection_stgcn/{timestamp}'
    
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='模型保存目录')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载checkpoint路径')
    
    args = parser.parse_args()
    return args


def save_training_summary(args, train_results, test_results, save_dir):
    """
    保存训练参数和结果摘要为JSON格式
    
    Args:
        args: 训练参数
        train_results: 训练过程中的最佳结果
        test_results: 测试集评估结果
        save_dir: 保存目录
    """
    # 提取需要保存的参数
    params = {
        "data_params": {
            "data_dir": args.data_dir,
            "normal_seq_length": args.normal_seq_length,
            "normal_stride": args.normal_stride,
            "fall_seq_length": args.fall_seq_length,
            "fall_stride": args.fall_stride,
            "overlap_threshold": args.overlap_threshold,
            "test_ratio": args.test_ratio,
            "val_ratio": args.val_ratio
        },
        "model_params": {
            "model_type": "ST-GCN",
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


def main():
    """
    主函数
    """
    args = process_args()
    
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
    
    # 创建以时间戳命名的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"创建保存目录: {save_dir}")
    
    # 创建数据集和加载器
    print("创建数据集...")
    full_dataset = PoseSequenceDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        normal_seq_length=args.normal_seq_length,
        normal_stride=args.normal_stride,
        fall_seq_length=args.fall_seq_length,
        fall_stride=args.fall_stride,
        overlap_threshold=args.overlap_threshold
    )
    
    print(f"数据集大小: {len(full_dataset)}")
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=args.test_ratio,
        stratify=[full_dataset[i]['label'].item() for i in range(len(full_dataset))],
        random_state=args.seed
    )
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=args.val_ratio,
        stratify=[full_dataset[i]['label'].item() for i in train_indices],
        random_state=args.seed
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 获取样本数据以确定输入维度
    sample_item = full_dataset[0]
    sequence_shape = sample_item['sequence'].shape
    
    # 计算关节数量和输入特征维度
    num_joints = 17  # COCO关键点数量
    in_channels = 3  # 每个关键点的坐标维度(x,y,z)
    
    print(f"序列形状: {sequence_shape}")
    print(f"关节数量: {num_joints}")
    print(f"输入通道数: {in_channels}")
    
    # 创建模型
    model = FallDetectionSTGCN(
        num_joints=num_joints,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
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
    
    # 仅测试模式
    if args.test_only:
        if args.checkpoint is None:
            print("错误: 测试模式需要提供checkpoint")
            return
        
        print("开始模型评估...")
        test_results = evaluate_model(model, test_loader, criterion, device, save_dir)
        
        # 保存测试结果摘要
        summary = save_training_summary(
            args=args,
            train_results={"val_loss": float('inf'), "val_acc": 0.0, "epoch": 0},
            test_results=test_results,
            save_dir=save_dir
        )
        print("评估完成!")
        return
    
    # 训练模型
    print("开始训练模型...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        scheduler=scheduler,
        save_dir=save_dir
    )
    print("训练完成!")
    
    # 加载最佳模型进行测试
    best_model_path = os.path.join(save_dir, 'best_model.pth')
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
    test_results = evaluate_model(model, test_loader, criterion, device, save_dir)
    
    # 保存训练摘要
    summary = save_training_summary(
        args=args,
        train_results=best_val_results,
        test_results=test_results,
        save_dir=save_dir
    )
    print("评估完成!")
    
    return model, history, test_results, summary


if __name__ == "__main__":
    main()
