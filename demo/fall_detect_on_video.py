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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import json

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
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 分类
        output = self.classifier(context)
        return output

# 导入项目中的其他模块
from demo.vis import get_pose2D, get_pose3D, visualize_pose2D, visualize_pose3D, generate_demo

# 定义PoseEvaluator类，避免从evaluate_pose导入
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
        # 模型输出高概率表示"正常"，所以跌倒概率 = 1 - 输出概率
        fall_probs = [1 - p for p in segment_probs]
        segment_preds = [1 if p >= 0.5 else 0 for p in fall_probs]
        
        # 计算整体预测结果
        # 策略: 如果任何一段预测为跌倒,则认为整个序列发生跌倒
        final_pred = 1 if any(segment_preds) else 0
        
        # 计算整体跌倒概率
        # 策略: 使用最高的跌倒概率
        final_prob = max(fall_probs)
        
        # 生成带帧范围的结果
        segment_results = []
        for i, (pred, prob) in enumerate(zip(segment_preds, fall_probs)):
            start_frame = i * self.seq_length
            end_frame = min((i + 1) * self.seq_length, total_frames)
            segment_results.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'prediction': pred,
                'fall_probability': prob  # 直接使用跌倒概率
            })
        
        return final_pred, final_prob, segment_results

class FallDetectionOnVideo:
    """视频跌倒检测类
    
    将视频转换为3D姿态并进行跌倒检测，同时生成可视化结果
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
        
        # 1. 生成2D姿态
        print(f"⏳ 从视频生成2D姿态...")
        keypoints = get_pose2D(self.args.video, self.output_dir, 
                              detector=self.args.detector, 
                              batch_size=self.args.batch_size)
        
        # 2. 生成3D姿态
        print(f"⏳ 从2D姿态预测3D姿态...")
        output_3d_data = get_pose3D(self.args.video, self.output_dir, self.args.fix_z)
        output_3d_file = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
        
        # 3. 可视化2D姿态
        print(f"⏳ 可视化2D姿态...")
        visualize_pose2D(self.args.video, self.output_dir, keypoints)
        
        # 4. 可视化3D姿态
        print(f"⏳ 可视化3D姿态...")
        visualize_pose3D(self.args.video, self.output_dir, self.args.fix_z, output_3d_data)
        
        # 5. 生成演示视频
        print(f"⏳ 生成演示视频...")
        generate_demo(self.output_dir)
        
        # 6. 进行跌倒检测
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
            
            # 7. 生成带跌倒检测结果的视频
            self.generate_fall_detection_video(segment_results)
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        minutes, seconds = divmod(total_time, 60)
        print(f"\n✓ 处理完成! 总耗时: {int(minutes)}分{seconds:.1f}秒")
    
    def generate_fall_detection_video(self, segment_results):
        """生成带跌倒检测结果的视频
        
        Args:
            segment_results: 每段的检测结果
        """
        print(f"⏳ 生成带跌倒检测结果的视频...")
        
        # 读取原始演示视频
        demo_dir = os.path.join(self.output_dir, 'pose')
        if not os.path.exists(demo_dir):
            print(f"错误: 未找到演示图像目录 {demo_dir}")
            return
        
        # 获取所有演示图像
        demo_images = sorted(os.listdir(demo_dir))
        if not demo_images:
            print(f"错误: 演示图像目录为空 {demo_dir}")
            return
        
        # 创建输出目录
        output_dir = os.path.join(self.output_dir, 'fall_detection')
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取第一张图像的尺寸
        first_img = Image.open(os.path.join(demo_dir, demo_images[0]))
        width, height = first_img.size
        
        # 准备字体
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            # 如果找不到，使用默认字体
            font = ImageFont.load_default()
        
        # 处理每一帧
        for i, img_name in enumerate(tqdm(demo_images, desc="生成检测视频")):
            # 找到当前帧所在的段
            current_segment = None
            for segment in segment_results:
                if segment['start_frame'] <= i < segment['end_frame']:
                    current_segment = segment
                    break
            
            if current_segment is None:
                continue
            
            # 读取原始演示图像
            img_path = os.path.join(demo_dir, img_name)
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            
            # 准备文本
            is_fall = current_segment['prediction'] == 1
            status = "FALL DETECTED" if is_fall else "NORMAL"
            prob = current_segment['fall_probability']
            text = f"{status} (Probability: {prob:.4f})"
            
            # 设置文本颜色和背景
            text_color = (255, 0, 0) if is_fall else (0, 128, 0)  # 红色表示跌倒，绿色表示正常
            
            # 计算文本位置 (顶部居中)
            try:
                # 新版PIL使用textbbox
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_width = right - left
                text_height = bottom - top
            except AttributeError:
                # 如果textbbox不可用，尝试使用textlength
                try:
                    text_width = draw.textlength(text, font=font)
                    # 估计文本高度
                    text_height = font.getsize(text)[1]
                except (AttributeError, TypeError):
                    # 如果都不可用，使用估计值
                    text_width = len(text) * 15  # 粗略估计
                    text_height = 36
            
            position = ((width - text_width) // 2, 10)
            
            # 绘制半透明背景
            bg_left = position[0] - 10
            bg_top = position[1] - 5
            bg_right = position[0] + text_width + 10
            bg_bottom = position[1] + text_height + 5
            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], 
                          fill=(0, 0, 0, 128))
            
            # 绘制文本
            draw.text(position, text, font=font, fill=text_color)
            
            # 保存图像
            output_path = os.path.join(output_dir, img_name)
            img.save(output_path)
        
        # 将图像序列转换为视频
        self.images_to_video(output_dir, os.path.join(self.output_dir, 'fall_detection_result.mp4'))
        print(f"✓ 跌倒检测视频已保存至: {os.path.join(self.output_dir, 'fall_detection_result.mp4')}")
    
    def images_to_video(self, image_dir, output_video_path, fps=30):
        """将图像序列转换为视频
        
        Args:
            image_dir: 图像目录
            output_video_path: 输出视频路径
            fps: 帧率
        """
        images = sorted(os.listdir(image_dir))
        if not images:
            print(f"错误: 图像目录为空 {image_dir}")
            return
        
        # 读取第一张图像获取尺寸
        first_img_path = os.path.join(image_dir, images[0])
        img = cv2.imread(first_img_path)
        height, width, layers = img.shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 添加每一帧
        for image in tqdm(images, desc="创建视频"):
            img_path = os.path.join(image_dir, image)
            video.write(cv2.imread(img_path))
        
        # 释放资源
        video.release()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频跌倒检测')
    
    # 输入输出参数
    parser.add_argument('--video', type=str, required=True,
                      help='输入视频路径')
    parser.add_argument('--output_dir', type=str, default='./demo/output_detect/',
                      help='输出目录')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='跌倒检测模型目录路径')
    
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
    
    # 创建跌倒检测器并处理视频
    detector = FallDetectionOnVideo(args)
    detector.process()

if __name__ == "__main__":
    main()