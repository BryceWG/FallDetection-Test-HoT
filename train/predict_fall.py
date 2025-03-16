#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from train_lstm import FallDetectionLSTM

class FallPredictor:
    """
    跌倒预测器类
    用于加载训练好的模型并对输入的3D姿态数据进行跌倒检测
    """
    def __init__(self, model_path, hidden_dim=256, num_layers=3, dropout=0.3, device=None):
        """
        初始化跌倒预测器
        
        参数:
            model_path: 模型checkpoint路径
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            device: 运行设备(cuda/cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        input_dim = 51  # 17个关键点 * 3(x,y,z)
        self.model = FallDetectionLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 设置序列长度
        self.seq_length = 30
        print("模型加载完成")
        
    def preprocess_sequence(self, pose_sequence):
        """
        预处理姿态序列数据
        
        参数:
            pose_sequence: numpy数组, 形状为[seq_len, num_joints, 3]
            
        返回:
            处理后的tensor, 形状为[1, seq_len, num_joints * 3]
        """
        # 确保序列长度正确
        if len(pose_sequence) < self.seq_length:
            # 如果序列不够长,通过重复最后一帧进行填充
            padding_length = self.seq_length - len(pose_sequence)
            last_frame = pose_sequence[-1:]
            pose_sequence = np.concatenate([pose_sequence, np.repeat(last_frame, padding_length, axis=0)], axis=0)
        elif len(pose_sequence) > self.seq_length:
            # 如果序列太长,只取最后seq_length帧
            pose_sequence = pose_sequence[-self.seq_length:]
            
        # 展平关键点数据
        flattened_sequence = pose_sequence.reshape(self.seq_length, -1)
        
        # 特征标准化
        seq_min = flattened_sequence.min(axis=0, keepdims=True)
        seq_max = flattened_sequence.max(axis=0, keepdims=True)
        seq_range = seq_max - seq_min
        seq_range[seq_range == 0] = 1.0  # 避免除零
        flattened_sequence = (flattened_sequence - seq_min) / seq_range
        
        # 转换为tensor并添加batch维度
        sequence_tensor = torch.FloatTensor(flattened_sequence).unsqueeze(0)
        return sequence_tensor
        
    def predict(self, pose_data, threshold=0.5, window_size=30, stride=15):
        """
        对姿态序列进行跌倒预测
        
        参数:
            pose_data: numpy数组,形状为[frames, num_joints, 3]
            threshold: 跌倒判定阈值
            window_size: 滑动窗口大小
            stride: 滑动步长
            
        返回:
            predictions: 每个窗口的预测概率
            fall_windows: 检测到跌倒的窗口索引
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # 使用滑动窗口进行预测
            for start_idx in range(0, len(pose_data) - window_size + 1, stride):
                end_idx = start_idx + window_size
                sequence = pose_data[start_idx:end_idx]
                
                # 预处理序列
                sequence_tensor = self.preprocess_sequence(sequence)
                sequence_tensor = sequence_tensor.to(self.device)
                
                # 预测
                output = self.model(sequence_tensor)
                pred_prob = output.item()
                predictions.append({
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'probability': pred_prob
                })
        
        # 找出预测概率超过阈值的窗口
        fall_windows = [pred for pred in predictions if pred['probability'] >= threshold]
        
        return predictions, fall_windows
    
    def visualize_predictions(self, predictions, save_path=None):
        """
        可视化预测结果
        
        参数:
            predictions: predict()方法返回的预测结果
            save_path: 保存图像的路径(可选)
        """
        frame_indices = [pred['start_frame'] for pred in predictions]
        probabilities = [pred['probability'] for pred in predictions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(frame_indices, probabilities, 'b-', label='Fall Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.fill_between(frame_indices, probabilities, 0.5,
                        where=[p >= 0.5 for p in probabilities],
                        color='red', alpha=0.3, label='Fall Region')
        
        plt.xlabel('Start Frame')
        plt.ylabel('Probability')
        plt.title('Fall Detection Results')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"预测结果图像已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()


def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='使用训练好的LSTM模型进行跌倒检测')
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型checkpoint路径')
    parser.add_argument('--pose_file', type=str, required=True,
                      help='要预测的3D姿态数据文件路径(.npz格式)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='跌倒判定阈值')
    parser.add_argument('--window_size', type=int, default=30,
                      help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=15,
                      help='滑动步长')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout比例')
    parser.add_argument('--output_dir', type=str, default='./fall_detection_results',
                      help='结果保存目录')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU ID')
    
    args = parser.parse_args()
    return args


def main():
    """
    主函数
    """
    args = process_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载3D姿态数据
        if not os.path.exists(args.pose_file):
            raise FileNotFoundError(f"找不到姿态数据文件: {args.pose_file}")
            
        pose_data = np.load(args.pose_file, allow_pickle=True)['reconstruction']
        print(f"加载姿态数据: shape={pose_data.shape}")
        
        # 创建预测器
        predictor = FallPredictor(
            model_path=args.model_path,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device
        )
        
        # 进行预测
        print("开始预测...")
        predictions, fall_windows = predictor.predict(
            pose_data,
            threshold=args.threshold,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # 保存预测结果
        results = {
            'predictions': predictions,
            'fall_windows': fall_windows,
            'model_path': args.model_path,
            'pose_file': args.pose_file,
            'threshold': args.threshold,
            'window_size': args.window_size,
            'stride': args.stride
        }
        
        # 保存为npz格式
        output_file = os.path.join(args.output_dir, 'prediction_results.npz')
        np.savez(output_file, **results)
        print(f"预测结果已保存至: {output_file}")
        
        # 保存为CSV格式
        csv_file = os.path.join(args.output_dir, 'prediction_results.csv')
        with open(csv_file, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("start_frame,end_frame,probability,is_fall\n")
            # 写入每个预测结果
            for pred in predictions:
                is_fall = 1 if pred['probability'] >= args.threshold else 0
                f.write(f"{pred['start_frame']},{pred['end_frame']},{pred['probability']:.4f},{is_fall}\n")
        print(f"预测结果已保存至CSV: {csv_file}")
        
        # 可视化结果
        vis_file = os.path.join(args.output_dir, 'prediction_visualization.png')
        predictor.visualize_predictions(predictions, save_path=vis_file)
        
        # 打印检测到的跌倒区间
        if fall_windows:
            print("\n检测到的跌倒区间:")
            for window in fall_windows:
                print(f"帧 {window['start_frame']} - {window['end_frame']}, "
                      f"跌倒概率: {window['probability']:.4f}")
        else:
            print("\n未检测到跌倒")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 