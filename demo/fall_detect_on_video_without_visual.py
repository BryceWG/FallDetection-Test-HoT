#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import json
from vis import get_pose2D, get_pose3D  # 只保留姿态提取相关函数
import glob  # 添加glob模块用于文件搜索
import pandas as pd  # 添加pandas用于读取CSV
from sklearn.metrics import confusion_matrix, classification_report  # 用于计算混淆矩阵
import seaborn as sns  # 用于绘制混淆矩阵
import matplotlib.pyplot as plt  # 用于绘图

# 添加项目根目录到路径
sys.path.append(os.getcwd())

# 直接定义FallDetectionLSTM类，避免从train_lstm导入
class FallDetectionLSTM(nn.Module):
    """基于LSTM的跌倒检测模型
    
    输入: 姿态序列数据 [batch_size, sequence_length, feature_dim]
    输出: 二分类结果 [batch_size, 1]
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(FallDetectionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 单向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 改为单向LSTM
        )
        
        # 注意力机制 - 调整维度以适应单向LSTM
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # 输入维度减半
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 分类层 - 简化结构并添加BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和得到上下文向量
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim]
        
        # 分类
        output = self.classifier(context)  # [batch_size, 1]
        
        return output

# 定义PoseEvaluator类，避免从evaluate_pose导入
class PoseEvaluator:
    """姿态评估器
    
    用于对单个3D姿态文件进行跌倒检测推理
    """
    def __init__(self, model_dir, device='cuda', seq_length=30, stride=10):
        """
        初始化评估器
        
        Args:
            model_dir: 模型保存目录,包含训练配置json和模型权重文件
            device: 运行设备
            seq_length: 序列长度
            stride: 滑动窗口步长
        """
        self.model_dir = model_dir
        self.device = device
        self.seq_length = seq_length
        self.stride = stride
        
        # 加载训练配置和标准化参数
        self.load_config()
        
        # 初始化模型
        self.init_model()
        
    def load_config(self):
        """加载训练配置和标准化参数"""
        config_file = os.path.join(self.model_dir, 'training_summary.json')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"找不到配置文件: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        self.config = config
        self.model_params = config['parameters']['model_params']
        self.norm_params = config['parameters']['normalization_params']
        
        # 加载标准化参数
        if 'normalization_params' not in config['parameters']:
            raise ValueError("配置文件中缺少标准化参数,请确保使用新版训练脚本重新训练模型")
            
        self.global_mean = np.array(self.norm_params['mean'])
        self.global_std = np.array(self.norm_params['std'])
        print(f"已加载标准化参数")
        print(f"全局均值范围: [{self.global_mean.min():.3f}, {self.global_mean.max():.3f}]")
        print(f"全局标准差范围: [{self.global_std.min():.3f}, {self.global_std.max():.3f}]")
        
    def init_model(self):
        """初始化模型"""
        # 创建模型
        self.model = FallDetectionLSTM(
            input_dim=17*3,  # 17个关键点 * 3维坐标
            hidden_dim=self.model_params['hidden_dim'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout']
        ).to(self.device)
        
        # 加载模型权重
        checkpoint_path = os.path.join(self.model_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型权重: {checkpoint_path}")
        
        # 设置为评估模式
        self.model.eval()
        
    def create_sequences(self, pose_data):
        """将姿态数据分割成序列
        
        Args:
            pose_data: 原始姿态数据 [N, 17, 3]
            
        Returns:
            sequences: 序列数据 [num_sequences, seq_length, 51]
            frame_indices: 每个序列对应的帧范围列表 [(start, end), ...]
        """
        sequences = []
        frame_indices = []
        total_frames = len(pose_data)
        
        for start_idx in range(0, total_frames - self.seq_length + 1, self.stride):
            end_idx = start_idx + self.seq_length
            if end_idx > total_frames:
                break
                
            # 获取序列数据并展平
            sequence = pose_data[start_idx:end_idx]
            flattened_sequence = sequence.reshape(self.seq_length, -1)
            sequences.append(flattened_sequence)
            
            # 记录对应的帧索引
            frame_indices.append((start_idx, end_idx))
        
        return np.array(sequences), frame_indices
        
    def normalize_sequences(self, sequences):
        """标准化序列数据"""
        return (sequences - self.global_mean) / self.global_std
        
    def predict_sequences(self, sequences, batch_size=32):
        """对序列进行预测
        
        Args:
            sequences: 标准化后的序列数据 [num_sequences, seq_length, 51]
            batch_size: 批处理大小
            
        Returns:
            预测结果数组 [num_sequences]
        """
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                outputs = self.model(batch_tensor)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
        
    def evaluate(self, pose_file, reverse=False):
        """评估单个姿态文件
        
        Args:
            pose_file: 3D姿态.npz文件路径
            reverse: 是否反转输入数据的时间顺序
            
        Returns:
            预测结果、跌倒概率和每段的预测结果
        """
        # 加载姿态数据
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"找不到姿态文件: {pose_file}")
            
        pose_data = np.load(pose_file, allow_pickle=True)['reconstruction']
        
        # 如果需要，反转时间顺序
        if reverse:
            pose_data = pose_data[::-1]
            print("已反转输入数据的时间顺序")
            
        total_frames = len(pose_data)
        
        # 创建序列
        sequences, frame_indices = self.create_sequences(pose_data)
        
        # 标准化数据
        normalized_sequences = self.normalize_sequences(sequences)
        
        # 进行预测
        predictions = self.predict_sequences(normalized_sequences)
        
        # 计算每帧的平均预测值
        all_frames = list(range(total_frames))
        all_predictions = np.zeros(len(all_frames))
        frame_counts = np.zeros(len(all_frames))
        
        # 累积每个序列的预测结果
        for (start, end), pred in zip(frame_indices, predictions):
            all_predictions[start:end] += pred.flatten()
            frame_counts[start:end] += 1
        
        # 计算平均预测值
        mask = frame_counts > 0
        all_predictions[mask] /= frame_counts[mask]
        
        # 生成每段的预测结果
        segment_results = []
        for (start, end), pred in zip(frame_indices, predictions):
            segment_results.append({
                'start_frame': start,
                'end_frame': end,
                'prediction': 1 if pred >= 0.5 else 0,
                'fall_probability': float(pred)
            })
        
        # 计算整体预测结果
        # 策略: 如果任何一段的平均预测值超过阈值,则认为发生跌倒
        final_pred = 1 if np.any(all_predictions >= 0.5) else 0
        final_prob = float(np.max(all_predictions))  # 使用最高的跌倒概率
        
        return final_pred, final_prob, segment_results

class FallDetectionOnVideo:
    """视频跌倒检测类
    
    将视频转换为3D姿态并进行跌倒检测
    """
    def __init__(self, args):
        """
        初始化视频跌倒检测
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置输出目录
        self.video_name = os.path.splitext(os.path.basename(args.video))[0]
        self.output_dir = os.path.join(args.output_dir, f"{self.video_name}/")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化评估器
        if args.model_dir:
            self.evaluator = PoseEvaluator(args.model_dir, self.device)
            print(f"已加载跌倒检测模型: {args.model_dir}")
        else:
            self.evaluator = None
            print("警告: 未指定模型目录，将不进行跌倒检测")
    
    def process(self):
        """处理视频并进行跌倒检测"""
        start_time = time.time()
        
        # 检查3D姿态文件是否已存在
        output_3d_file = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
        if os.path.exists(output_3d_file):
            print(f"发现已存在的3D姿态文件: {output_3d_file}")
            print("跳过姿态提取步骤...")
        else:
            # 1. 生成2D姿态
            print(f"⏳ 从视频生成2D姿态...")
            keypoints = get_pose2D(self.args.video, self.output_dir, 
                                  detector=self.args.detector, 
                                  batch_size=self.args.batch_size)
            
            # 2. 生成3D姿态
            print(f"⏳ 从2D姿态预测3D姿态...")
            output_3d_data = get_pose3D(self.args.video, self.output_dir, self.args.fix_z)
        
        # 3. 进行跌倒检测
        if self.evaluator:
            print(f"⏳ 进行跌倒检测...")
            pred, prob, segment_results = self.evaluator.evaluate(output_3d_file, reverse=self.args.reverse)
            
            # 输出检测结果
            result = "跌倒" if pred == 1 else "正常"
            print(f"\n评估结果:")
            print(f"文件: {output_3d_file}")
            print(f"预测: {result}")
            print(f"跌倒概率: {prob:.4f}")
            
            # 输出每段的预测结果
            print(f"\n分段预测结果:")
            for i, result in enumerate(segment_results):
                seg_result = "跌倒" if result['prediction'] == 1 else "正常"
                print(f"段 {i+1} (帧 {result['start_frame']}-{result['end_frame']}): "
                      f"{seg_result} (跌倒概率: {result['fall_probability']:.4f})")
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        minutes, seconds = divmod(total_time, 60)
        print(f"\n✓ 处理完成! 总耗时: {int(minutes)}分{seconds:.1f}秒")

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        output_dir: 输出目录
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fall'],
                yticklabels=['Normal', 'Fall'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 保存图形
    output_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_file)
    plt.close()
    print(f"混淆矩阵已保存至: {output_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频跌倒检测')
    
    # 输入输出参数
    parser.add_argument('--video', type=str, default=None,
                      help='输入视频路径')
    parser.add_argument('--video_dir', type=str, default=None,
                      help='输入视频文件夹路径，当需要处理多个视频时使用')
    parser.add_argument('--output_dir', type=str, default='./demo/output_detect/',
                      help='输出目录')
    parser.add_argument('--model_dir', type=str, default='checkpoint/fall_detection_lstm/2025-03-22-2328-b',
                      help='跌倒检测模型目录路径')
    parser.add_argument('--ground_truth', type=str, default=None,
                      help='标注数据CSV文件路径')
    
    # 硬件参数
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU ID')
    
    # 姿态估计参数
    parser.add_argument('--fix_z', action='store_true',
                      help='固定Z轴')
    parser.add_argument('--detector', type=str, default='yolo11', 
                      choices=['yolov3', 'yolo11'],
                      help='选择人体检测器')
    parser.add_argument('--batch_size', type=int, default=2000,
                      help='帧分组大小，默认为2000帧')
    
    # 跌倒检测参数
    parser.add_argument('--reverse', action='store_true',
                      help='反转输入数据的时间顺序')
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 检查输入参数
    if args.video is None and args.video_dir is None:
        raise ValueError("请提供单个视频路径(--video)或视频文件夹路径(--video_dir)")
    
    if args.video is not None and args.video_dir is not None:
        print("警告: 同时提供了单个视频和视频文件夹路径，将优先处理视频文件夹")
    
    # 读取标注数据（如果提供）
    ground_truth = None
    if args.ground_truth:
        if not os.path.exists(args.ground_truth):
            print(f"警告: 找不到标注文件 '{args.ground_truth}'")
        else:
            ground_truth = pd.read_csv(args.ground_truth)
            print(f"已加载标注数据，共 {len(ground_truth)} 条记录")
    
    # 处理单个视频
    if args.video_dir is None and args.video is not None:
        detector = FallDetectionOnVideo(args)
        detector.process()
        return
    
    # 处理视频文件夹
    video_dir = args.video_dir
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    # 收集所有视频文件
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if len(video_files) == 0:
        print(f"错误: 在目录 '{video_dir}' 中未找到任何视频文件")
        return
    
    print(f"在目录 '{video_dir}' 中找到 {len(video_files)} 个视频文件")
    for i, video_file in enumerate(video_files):
        print(f"{i+1}. {os.path.basename(video_file)}")
    
    # 处理每个视频文件
    results = []
    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    
    for i, video_file in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] 处理视频: {os.path.basename(video_file)}")
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # 更新args中的视频路径
        args.video = video_file
        detector = FallDetectionOnVideo(args)
        
        # 记录开始时间
        video_start_time = time.time()
        detector.process()
        
        # 记录结束时间和结果
        video_end_time = time.time()
        video_total_time = video_end_time - video_start_time
        minutes, seconds = divmod(video_total_time, 60)
        
        result_info = {
            'video_name': video_name,
            'processing_time': f"{int(minutes)}分{seconds:.1f}秒",
            'processing_time_seconds': video_total_time  # 添加处理时间（秒）
        }
        
        if detector.evaluator:
            output_3d_file = os.path.join(detector.output_dir, 'output_3D', 'output_keypoints_3d.npz')
            pred, prob, segment_results = detector.evaluator.evaluate(output_3d_file, reverse=args.reverse)
            result = "跌倒" if pred == 1 else "正常"
            result_info['prediction'] = result
            result_info['prediction_value'] = int(pred)  # 添加数值预测结果
            result_info['fall_probability'] = f"{prob:.4f}"
            result_info['fall_probability_value'] = float(prob)  # 添加数值概率
            result_info['segment_results'] = segment_results  # 添加分段结果
            
            # 如果有标注数据，进行对比
            if ground_truth is not None:
                gt_row = ground_truth[ground_truth['video_id'] == video_name]
                if not gt_row.empty:
                    gt_has_fall = gt_row.iloc[0]['has_fall']
                    result_info['ground_truth'] = "跌倒" if gt_has_fall == 1 else "正常"
                    result_info['ground_truth_value'] = int(gt_has_fall)  # 添加数值真实标签
                    result_info['is_correct'] = pred == gt_has_fall
                    
                    # 记录用于混淆矩阵
                    y_true.append(gt_has_fall)
                    y_pred.append(pred)
                else:
                    result_info['ground_truth'] = "未知"
                    result_info['ground_truth_value'] = None
                    result_info['is_correct'] = None
        
        results.append(result_info)
    
    # 保存处理结果到JSON文件
    json_output_path = os.path.join(args.output_dir, 'detection_results.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✓ 处理结果已保存到: {json_output_path}")
    
    # 打印所有视频的处理结果汇总
    print("\n" + "="*60)
    print("所有视频处理结果汇总:")
    print("="*60)
    
    correct_count = 0
    total_count = 0
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['video_name']}:")
        print(f"   处理时间: {result['processing_time']}")
        if 'prediction' in result:
            print(f"   预测结果: {result['prediction']}")
            print(f"   跌倒概率: {result['fall_probability']}")
            if 'ground_truth' in result:
                print(f"   真实标签: {result['ground_truth']}")
                if result['is_correct'] is not None:
                    status = "✓" if result['is_correct'] else "✗"
                    print(f"   判断结果: {status}")
                    if result['is_correct']:
                        correct_count += 1
                    total_count += 1
        print("-"*60)
    
    # 如果有标注数据，计算并显示评估指标
    if ground_truth is not None and len(y_true) > 0:
        print("\n性能评估:")
        print("="*60)
        
        # 计算准确率
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"准确率: {accuracy:.2%} ({correct_count}/{total_count})")
        
        # 打印分类报告
        report = classification_report(y_true, y_pred, target_names=['正常', '跌倒'])
        print("\n详细评估指标:")
        print(report)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_true, y_pred, args.output_dir)
        
        # 保存评估指标到JSON文件
        metrics = {
            'accuracy': float(accuracy),
            'correct_count': correct_count,
            'total_count': total_count,
            'classification_report': classification_report(y_true, y_pred, target_names=['正常', '跌倒'], output_dict=True)
        }
        metrics_json_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"\n✓ 评估指标已保存到: {metrics_json_path}")

if __name__ == "__main__":
    main()