#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python train_lstm.py --label_file fall_frame_data.csv --data_dir ./demo/output/ --seq_length 30 --stride 15 --batch_size 8 --num_epochs 100 --learning_rate 0.0005 --hidden_dim 256 --num_layers 3 --dropout 0.3 --save_dir ./my_models/fall_detection/ 
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

sys.path.append(os.getcwd())


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
    def __init__(self, data_dir, label_file, seq_length=60, stride=30, transform=None, test_mode=False):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        self.test_mode = test_mode
        
        # 加载标签数据
        self.labels_df = pd.read_csv(label_file)
        
        # 创建数据索引
        self._create_sequence_samples()
        
    def _create_sequence_samples(self):
        """
        创建序列样本索引
        - 非跌倒视频: 序列长度30帧,滑动步长20,全部标记为非跌倒(0)
        - 跌倒视频: 序列长度30帧,滑动步长15,根据与跌倒帧段重叠比例判断标签
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
                # 非跌倒视频: 使用较大的滑动步长(20)
                for start_idx in range(0, total_frames - self.seq_length + 1, 20):
                    end_idx = start_idx + self.seq_length
                    
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
                # 跌倒视频: 使用较小的滑动步长(15)
                fall_start = int(row['fall_start_frame'])
                fall_end = int(row['fall_end_frame'])
                fall_duration = fall_end - fall_start + 1
                
                for start_idx in range(0, total_frames - self.seq_length + 1, 15):
                    end_idx = start_idx + self.seq_length
                    
                    # 如果剩余帧数不足一个完整序列,则跳过
                    if end_idx > total_frames:
                        break
                    
                    # 计算当前序列与跌倒帧段的重叠部分
                    overlap_start = max(start_idx, fall_start)
                    overlap_end = min(end_idx, fall_end)
                    
                    if overlap_end > overlap_start:
                        # 存在重叠,计算重叠比例
                        overlap_length = overlap_end - overlap_start + 1
                        overlap_ratio = overlap_length / self.seq_length
                        
                        # 如果重叠比例超过30%,标记为跌倒
                        is_fall = overlap_ratio >= 0.3
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
        if end_idx - start_idx < self.seq_length:
            # 如果序列不够长,进行填充
            padding_length = self.seq_length - (end_idx - start_idx)
            sequence = pose_data[start_idx:end_idx]
            
            # 通过重复最后一帧进行填充
            last_frame = sequence[-1:].repeat(padding_length, axis=0)
            sequence = np.concatenate([sequence, last_frame], axis=0)
        else:
            # 如果序列足够长,直接截取指定长度
            sequence = pose_data[start_idx:start_idx + self.seq_length]
        
        # 展平每一帧的关键点数据
        # 原始形状: [seq_len, num_joints, 3]
        # 转换为: [seq_len, num_joints * 3]
        num_joints = sequence.shape[1]
        flattened_sequence = sequence.reshape(self.seq_length, -1)
        
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
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='训练损失')
    plt.plot(history['val_losses'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
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


def evaluate_model(model, test_loader, criterion, device, save_dir='./checkpoints/fall_detection'):
    """
    评估模型
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
    
    # 二值化预测结果
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
    
    # 在格子中显示数字
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
    parser = argparse.ArgumentParser(description='跌倒检测模型训练')
    parser.add_argument('--data_dir', type=str, default='./demo/output/', help='3D姿态数据目录')
    parser.add_argument('--label_file', type=str, required=True, help='标签文件路径(CSV格式)')
    parser.add_argument('--seq_length', type=int, default=30, help='序列长度')
    parser.add_argument('--stride', type=int, default=15, help='滑动窗口步长')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/fall_detection_lstm/', help='模型保存目录')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载checkpoint路径')
    
    args = parser.parse_args()
    return args


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
    
    # 创建数据集和加载器
    print("创建数据集...")
    full_dataset = PoseSequenceDataset(
        data_dir=args.data_dir,
        label_file=args.label_file,
        seq_length=args.seq_length,
        stride=args.stride
    )
    
    print(f"数据集大小: {len(full_dataset)}")
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=[full_dataset[i]['label'].item() for i in range(len(full_dataset))],
        random_state=args.seed
    )
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.25,  # 验证集占训练集的25%
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
        results = evaluate_model(model, test_loader, criterion, device, args.save_dir)
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
        save_dir=args.save_dir
    )
    print("训练完成!")
    
    # 评估模型
    print("开始模型评估...")
    results = evaluate_model(model, test_loader, criterion, device, args.save_dir)
    print("评估完成!")
    
    return model, history, results
    

if __name__ == "__main__":
    main()