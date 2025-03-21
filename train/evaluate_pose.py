#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from train.train_lstm import FallDetectionLSTM

class PoseEvaluator:
    """姿态评估器
    
    用于对单个3D姿态文件进行跌倒检测推理
    """
    def __init__(self, model_dir, device='cuda'):
        """
        初始化评估器
        
        Args:
            model_dir: 模型保存目录,包含训练配置json和模型权重文件
            device: 运行设备
        """
        self.model_dir = model_dir
        self.device = device
        
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
        params = config['parameters']
        
        # 加载序列长度
        self.seq_length = params['data_params']['normal_seq_length']
        print(f"已加载配置文件: {config_file}")
        print(f"序列长度: {self.seq_length}")
        
        # 加载标准化参数
        if 'normalization_params' not in params:
            raise ValueError("配置文件中缺少标准化参数,请确保使用新版训练脚本重新训练模型")
            
        norm_params = params['normalization_params']
        self.global_mean = np.array(norm_params['mean'])
        self.global_std = np.array(norm_params['std'])
        print(f"已加载标准化参数")
        print(f"全局均值范围: [{self.global_mean.min():.3f}, {self.global_mean.max():.3f}]")
        print(f"全局标准差范围: [{self.global_std.min():.3f}, {self.global_std.max():.3f}]")
        
    def init_model(self):
        """初始化模型"""
        # 加载模型参数
        model_params = self.config['parameters']['model_params']
        
        # 创建模型
        self.model = FallDetectionLSTM(
            input_dim=17*3,  # 17个关键点 * 3维坐标
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout']
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
        
    def preprocess_pose(self, pose_data):
        """预处理姿态数据
        
        Args:
            pose_data: 原始姿态数据 [N, 17, 3]
            
        Returns:
            处理后的数据 [num_segments, seq_length, 51]
        """
        total_frames = len(pose_data)
        num_segments = (total_frames + self.seq_length - 1) // self.seq_length
        segments = []
        
        for i in range(num_segments):
            start_idx = i * self.seq_length
            end_idx = min((i + 1) * self.seq_length, total_frames)
            
            # 获取当前段
            segment = pose_data[start_idx:end_idx]
            
            # 如果是最后一段且长度不足,通过重复最后一帧进行填充
            if len(segment) < self.seq_length:
                padding_length = self.seq_length - len(segment)
                padding = np.tile(segment[-1:], (padding_length, 1, 1))
                segment = np.concatenate([segment, padding], axis=0)
            
            # 展平每一帧的关键点
            segment = segment.reshape(self.seq_length, -1)
            
            # 应用Z-score标准化
            segment = (segment - self.global_mean) / self.global_std
            
            segments.append(segment)
            
        # 将所有段堆叠成batch
        segments = np.stack(segments, axis=0)
        return torch.FloatTensor(segments)
        
    def evaluate(self, pose_file):
        """评估单个姿态文件
        
        Args:
            pose_file: 3D姿态.npz文件路径
            
        Returns:
            预测结果、置信度和每段的预测结果
        """
        # 加载姿态数据
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"找不到姿态文件: {pose_file}")
            
        pose_data = np.load(pose_file, allow_pickle=True)['reconstruction']
        total_frames = len(pose_data)
        
        # 预处理数据
        pose_tensor = self.preprocess_pose(pose_data)
        pose_tensor = pose_tensor.to(self.device)
        
        # 进行推理
        segment_probs = []
        with torch.no_grad():
            # 如果数据量太大,分批处理
            batch_size = 32
            for i in range(0, len(pose_tensor), batch_size):
                batch = pose_tensor[i:i + batch_size]
                outputs = self.model(batch)
                segment_probs.extend(outputs.cpu().numpy().flatten())
        
        # 获取每段的预测结果
        segment_preds = [1 if p >= 0.5 else 0 for p in segment_probs]
        
        # 计算整体预测结果
        # 策略: 如果任何一段预测为跌倒,则认为整个序列发生跌倒
        final_pred = 1 if any(segment_preds) else 0
        
        # 计算整体置信度
        # 策略: 使用最高的跌倒概率作为整体置信度
        final_prob = max(segment_probs) if final_pred == 1 else 1 - max(segment_probs)
        
        # 生成带帧范围的结果
        segment_results = []
        for i, (pred, prob) in enumerate(zip(segment_preds, segment_probs)):
            start_frame = i * self.seq_length
            end_frame = min((i + 1) * self.seq_length, total_frames)
            segment_results.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'prediction': pred,
                'probability': prob
            })
        
        return final_pred, final_prob, segment_results
        
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='3D姿态跌倒检测评估')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='模型目录路径,包含训练配置json和模型权重文件')
    parser.add_argument('--pose_file', type=str, required=True,
                      help='要评估的3D姿态.npz文件路径')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU ID')
    args = parser.parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 创建评估器
        evaluator = PoseEvaluator(args.model_dir, device)
        
        # 进行评估
        pred, prob, segment_results = evaluator.evaluate(args.pose_file)
        
        # 输出结果
        result = "跌倒" if pred == 1 else "正常"
        print(f"\n评估结果:")
        print(f"文件: {args.pose_file}")
        print(f"预测: {result}")
        print(f"置信度: {prob:.4f}")
        
        # 输出每段的预测结果
        print(f"\n分段预测结果:")
        for i, result in enumerate(segment_results):
            seg_result = "跌倒" if result['prediction'] == 1 else "正常"
            print(f"段 {i+1} (帧 {result['start_frame']}-{result['end_frame']}): "
                  f"{seg_result} (置信度: {result['probability']:.4f})")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
