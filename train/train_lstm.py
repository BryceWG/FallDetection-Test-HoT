#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python train/train_lstm.py --label_file fall_frame_data.csv
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
from collections import Counter
from data_balancer import create_balanced_loader

sys.path.append(os.getcwd())


class EarlyStopping:
    """早停策略
    
    当验证集性能在patience个epoch内没有改善时，提前结束训练
    
    Args:
        patience (int): 等待改善的epoch数
        min_delta (float): 最小改善幅度，小于此值视为没有改善
        verbose (bool): 是否打印早停信息
    """
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
    """
    基于LSTM的跌倒检测模型
    输入: 姿态序列数据 [batch_size, sequence_length, feature_dim]
    输出: 二分类结果 [batch_size, 1]
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(FallDetectionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权平均
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # 分类预测
        output = self.classifier(context_vector)
        
        return output


class PoseSequenceDataset(Dataset):
    """
    姿态序列数据集
    加载3D姿态数据和对应的标签
    """
    def __init__(self, data_dir, label_file, normal_seq_length=30, normal_stride=20,
                 fall_seq_length=30, fall_stride=15, transform=None, test_mode=False):
        self.data_dir = data_dir
        self.normal_seq_length = normal_seq_length
        self.normal_stride = normal_stride
        self.fall_seq_length = fall_seq_length
        self.fall_stride = fall_stride
        self.transform = transform
        self.test_mode = test_mode
        
        # 加载标签数据
        self.labels_df = pd.read_csv(label_file)
        
        # 创建数据索引
        self._create_sequence_samples()
        self._compute_global_stats()
    
    def _load_pose_data(self, video_id, has_fall):
        """
        根据视频ID和是否包含跌倒加载对应的姿态数据
        """
        if has_fall:
            # 如果是跌倒视频,读取splits目录下的数据
            npz_file = os.path.join(self.data_dir, video_id, 
                                  'output_3D/splits/fall_keypoints_3d.npz')
        else:
            # 其他情况读取原始数据
            npz_file = os.path.join(self.data_dir, video_id, 
                                  'output_3D/output_keypoints_3d.npz')
            
        if not os.path.exists(npz_file):
            print(f"警告: 找不到文件 {npz_file}")
            return None
            
        return np.load(npz_file, allow_pickle=True)['reconstruction']
    
    def _create_sequence_samples(self):
        """
        创建序列样本索引
        """
        self.samples = []
        
        # 遍历所有视频
        for _, row in self.labels_df.iterrows():
            video_id = row['video_id']
            has_fall = int(row['has_fall'])
            
            # 加载姿态数据
            pose_data = self._load_pose_data(video_id, has_fall)
            if pose_data is None:
                continue
                
            total_frames = len(pose_data)
            
            if has_fall == 0:
                # 非跌倒视频：使用normal_stride
                for start_idx in range(0, total_frames - self.normal_seq_length + 1, 
                                     self.normal_stride):
                    end_idx = start_idx + self.normal_seq_length
                    if end_idx > total_frames:
                        break
                    
                    self.samples.append({
                        'video_id': video_id,
                        'label': has_fall,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
            else:
                # 跌倒视频：使用fall_stride
                for start_idx in range(0, total_frames - self.fall_seq_length + 1,
                                     self.fall_stride):
                    end_idx = start_idx + self.fall_seq_length
                    if end_idx > total_frames:
                        break
                    
                    self.samples.append({
                        'video_id': video_id,
                        'label': 1,  # 跌倒视频的所有片段都标记为跌倒
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
        
        # 打印数据集统计信息
        total_samples = len(self.samples)
        fall_samples = sum(1 for sample in self.samples if sample['label'] == 1)
        non_fall_samples = total_samples - fall_samples
        
        print(f"\n数据集统计信息:")
        print(f"总样本数: {total_samples}")
        print(f"跌倒样本数: {fall_samples}")
        print(f"非跌倒样本数: {non_fall_samples}")
        if total_samples > 0:
            print(f"跌倒/非跌倒比例: {fall_samples/non_fall_samples:.2f}")
    
    def _compute_global_stats(self):
        """
        计算所有训练数据的均值和标准差，用于Z-score标准化
        """
        print("\n计算全局统计信息...")
        all_sequences = []
        
        for sample in tqdm(self.samples, desc="加载数据"):
            pose_data = self._load_pose_data(sample['video_id'], sample['label'] == 1)
            if pose_data is None:
                continue
                
            start_idx = sample['start_idx']
            end_idx = sample['end_idx']
            
            sequence = pose_data[start_idx:end_idx]
            sequence = sequence.reshape(len(sequence), -1)
            all_sequences.append(sequence)
            
        all_data = np.concatenate(all_sequences, axis=0)
        self.global_mean = np.mean(all_data, axis=0)
        self.global_std = np.std(all_data, axis=0)
        # 避免除零
        self.global_std[self.global_std < 1e-7] = 1.0
        
        print(f"全局均值范围: [{self.global_mean.min():.3f}, {self.global_mean.max():.3f}]")
        print(f"全局标准差范围: [{self.global_std.min():.3f}, {self.global_std.max():.3f}]")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载3D姿态数据
        pose_data = self._load_pose_data(sample['video_id'], sample['label'] == 1)
        
        # 截取指定范围的序列
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # 确保序列长度一致
        if end_idx - start_idx < self.normal_seq_length:
            padding_length = self.normal_seq_length - (end_idx - start_idx)
            sequence = pose_data[start_idx:end_idx]
            last_frame = sequence[-1:].repeat(padding_length, axis=0)
            sequence = np.concatenate([sequence, last_frame], axis=0)
        else:
            sequence = pose_data[start_idx:start_idx + self.normal_seq_length]
        
        # 展平每一帧的关键点数据
        flattened_sequence = sequence.reshape(self.normal_seq_length, -1)
        
        # 使用Z-score标准化
        if self.transform:
            flattened_sequence = self.transform(flattened_sequence)
        else:
            flattened_sequence = (flattened_sequence - self.global_mean) / self.global_std
        
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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None, l1_lambda=0, save_dir='./checkpoints/fall_detection'):
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
            
            # 添加L1正则化
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
            
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
    parser.add_argument('--data_dir', type=str, default='./demo/output/', help='3D姿态数据目录')
    parser.add_argument('--label_file', type=str, required=False, help='标签文件路径(CSV格式)')
    
    # 非跌倒视频的序列参数
    parser.add_argument('--normal_seq_length', type=int, default=30, help='非跌倒视频序列长度')
    parser.add_argument('--normal_stride', type=int, default=10, help='非跌倒视频滑动步长')
    
    # 跌倒视频的序列参数
    parser.add_argument('--fall_seq_length', type=int, default=30, help='跌倒视频序列长度')
    parser.add_argument('--fall_stride', type=int, default=10, help='跌倒视频滑动步长')
    
    # 数据集划分参数
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集占总数据的比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集占训练数据的比例')
    
    # 数据平衡参数
    parser.add_argument('--balance_strategy', type=str, default='none', 
                       choices=['none', 'oversample', 'undersample', 'smote'],
                       help='数据平衡策略')
    parser.add_argument('--target_ratio', type=float, default=1.0,
                       help='目标类别比例(跌倒:非跌倒),仅在balance_strategy不为none时有效')
    
    # 其他原有参数
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2正则化系数')
    parser.add_argument('--l1_lambda', type=float, default=1e-5, help='L1正则化系数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 生成带时间戳的保存目录
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
    default_save_dir = f'./checkpoint/fall_detection_lstm/{timestamp}'
    
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
            "l1_lambda": args.l1_lambda,
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
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据集和加载器
    print("创建数据集...")
    full_dataset = PoseSequenceDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        normal_seq_length=args.normal_seq_length,
        normal_stride=args.normal_stride,
        fall_seq_length=args.fall_seq_length,
        fall_stride=args.fall_stride
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
    
    # 计算输入特征维度(姿态关键点数量 * 3)
    sample_item = full_dataset[0]
    input_dim = sample_item['sequence'].shape[1]
    print(f"输入特征维度: {input_dim}")
    
    # 创建模型
    model = FallDetectionLSTM(
        input_dim=input_dim,
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
        test_results = evaluate_model(model, test_loader, criterion, device, args.save_dir)
        
        # 保存测试结果摘要
        summary = save_training_summary(
            args=args,
            train_results={"val_loss": float('inf'), "val_acc": 0.0, "epoch": 0},
            test_results=test_results,
            save_dir=args.save_dir
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
        l1_lambda=args.l1_lambda,
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
        save_dir=args.save_dir
    )
    print("评估完成!")
    
    return model, history, test_results, summary
    

if __name__ == "__main__":
    main()