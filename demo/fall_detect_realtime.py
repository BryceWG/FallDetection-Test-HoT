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
from collections import deque
from queue import Queue
import threading
from scipy.interpolate import interp1d
from vis import get_pose2D, get_pose3D  # 只保留姿态提取相关函数

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

class RealtimeFallDetection:
    """实时视频跌倒检测类"""
    def __init__(self, args):
        """初始化实时跌倒检测"""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化队列和线程控制
        self.frame_queue = Queue()  # 移除maxsize限制
        self.result_queue = Queue()
        self.stop_flag = threading.Event()
        
        # 设置输出目录
        self.output_dir = os.path.join(args.output_dir, f"camera_{args.camera}/")
        self.temp_dir = os.path.join(self.output_dir, 'temp')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print("Initializing models...")
        
        # 加载人体检测模型
        from lib.yolo11.human_detector import load_model as yolo_model
        from lib.yolo11.human_detector import yolo_human_det as yolo_det
        from lib.yolo11.human_detector import reset_target
        reset_target()
            
        # 初始化评估器(保持模型常驻内存)
        if args.model_dir:
            self.evaluator = PoseEvaluator(args.model_dir, self.device)
            print(f"Model loaded: {args.model_dir}")
        else:
            self.evaluator = None
            print("Warning: No model specified")
            
        # 初始化结果显示变量
        self.current_status = "Normal"
        self.current_prob = 0.0
        self.status_update_time = 0
        
        # 预热模型
        print("Warming up models...")
        self._warmup_models()
        print("Models ready!")

    def _warmup_models(self):
        """预热模型，进行一次完整的处理流程"""
        # 创建一个全黑的测试帧
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frames = [test_frame] * 120  # 生成120帧测试数据
        
        print("预热模型中，这可能需要几秒钟...")
        # 静默处理预热帧
        with open(os.devnull, 'w') as f:
            # 临时重定向标准输出
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                self.process_video_segment(test_frames)
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout
        
        print("模型预热完成!")

    def process_video_segment(self, frames, need_interpolation=False):
        """处理视频片段"""
        try:
            # 保存为临时视频文件
            temp_video_path = os.path.join(self.temp_dir, f'temp_sequence_{time.time()}.mp4')
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            if os.path.exists(temp_video_path):
                # 生成2D姿态
                keypoints = get_pose2D(temp_video_path, self.output_dir,
                                     detector=self.args.detector, 
                                     batch_size=self.args.batch_size)
                
                if keypoints is not None and len(keypoints) > 0:
                    # 如果需要，进行线性插值
                    if need_interpolation:
                        keypoints = self.interpolate_keypoints(keypoints, self.args.buffer_size)
                    
                    # 生成3D姿态
                    output_3d_data = get_pose3D(temp_video_path, self.output_dir, self.args.fix_z)
                    output_3d_file = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
                    
                    # 进行跌倒检测
                    if self.evaluator and os.path.exists(output_3d_file):
                        pred, prob, _ = self.evaluator.evaluate(output_3d_file, reverse=False)
                        return pred, prob
            
            return None, None
            
        except Exception as e:
            print(f"Error processing video segment: {str(e)}")
            return None, None
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except:
                pass

    def process_frames_thread(self):
        """后台处理线程"""
        frames_buffer = []
        
        while not self.stop_flag.is_set():
            # 检查队列中的帧数
            pending_frames = self.frame_queue.qsize()
            
            # 收集帧直到达到缓冲区大小
            while len(frames_buffer) < self.args.buffer_size:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    frames_buffer.append(frame)
                except:
                    break
                    
            # 如果收集到足够的帧，进行处理
            if len(frames_buffer) >= self.args.buffer_size:
                # 检查是否需要自适应采样
                if pending_frames > self.args.max_pending_frames:
                    print(f"帧数堆积超过阈值({pending_frames}帧)，启用自适应采样，采样率: {self.args.sampling_rate}")
                    # 均匀采样帧
                    sampled_frames = self.adaptive_sampling(frames_buffer, self.args.sampling_rate)
                    pred, prob = self.process_video_segment(sampled_frames, need_interpolation=True)
                else:
                    pred, prob = self.process_video_segment(frames_buffer)
                
                if pred is not None:
                    self.result_queue.put((pred, prob, time.time()))
                frames_buffer = frames_buffer[int(self.args.buffer_size * (1 - self.args.overlap_ratio)):]  # 保留后半部分帧用于下次处理
                
            # 如果停止标志已设置且没有更多帧，退出循环
            if self.stop_flag.is_set() and self.frame_queue.empty():
                break

    def adaptive_sampling(self, frames, sampling_rate):
        """对帧进行自适应采样
        
        Args:
            frames: 输入帧列表
            sampling_rate: 采样率 (0-1)
            
        Returns:
            采样后的帧列表
        """
        total_frames = len(frames)
        sample_count = max(2, int(total_frames * sampling_rate))  # 至少保留2帧用于插值
        
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        return [frames[i] for i in indices]
    
    def interpolate_keypoints(self, keypoints, target_length):
        """对关键点进行线性插值
        
        Args:
            keypoints: 关键点数组 [n, 17, 2]
            target_length: 目标长度
            
        Returns:
            插值后的关键点数组 [target_length, 17, 2]
        """
        if len(keypoints) < 2:
            return keypoints
            
        # 创建插值函数
        frame_indices = np.arange(len(keypoints))
        target_indices = np.linspace(0, len(keypoints) - 1, target_length)
        
        # 对每个关键点进行插值
        interpolated = np.zeros((target_length, keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)
        
        for j in range(keypoints.shape[1]):  # 对每个关节
            for k in range(keypoints.shape[2]):  # 对x和y坐标
                # 创建插值函数
                interp_func = interp1d(frame_indices, keypoints[:, j, k], kind='linear', bounds_error=False, fill_value='extrapolate')
                # 应用插值
                interpolated[:, j, k] = interp_func(target_indices)
                
        return interpolated

    def display_frame(self, frame):
        """在帧上显示检测结果"""
        # 尝试获取最新的检测结果
        try:
            while not self.result_queue.empty():
                pred, prob, timestamp = self.result_queue.get_nowait()
                self.current_status = "Fall" if pred == 1 else "Normal"
                self.current_prob = prob
                self.status_update_time = timestamp
        except Exception as e:
            print(f"Error getting results: {str(e)}")
            
        # 计算显示区域的高度
        text_lines = 4  # 状态、概率、未处理帧数和采样状态
        rect_height = 30 * text_lines + 20
        
        # 添加半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 10 + rect_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 添加文本
        status_color = (0, 0, 255) if self.current_status == "Fall" else (0, 255, 0)
        cv2.putText(frame, f"Status: {self.current_status}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Probability: {self.current_prob:.2f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示未处理的帧数
        pending_frames = self.frame_queue.qsize()
        queue_color = (0, 255, 255) if pending_frames < self.args.max_pending_frames else (0, 0, 255)
        cv2.putText(frame, f"Pending Frames: {pending_frames}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, queue_color, 2)
        
        # 显示采样状态
        if pending_frames > self.args.max_pending_frames:
            cv2.putText(frame, f"Sampling Rate: {self.args.sampling_rate:.2f}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame

    def start_camera(self):
        """启动摄像头实时检测"""
        # 打开摄像头
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.args.camera}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 获取实际的摄像头参数
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera parameters: {frame_width}x{frame_height} @ {actual_fps}fps")
        
        # 创建输出视频写入器
        output_path = os.path.join(self.output_dir, 'realtime_detection.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, actual_fps, 
                            (frame_width, frame_height))
        
        # 启动处理线程
        process_thread = threading.Thread(target=self.process_frames_thread)
        process_thread.start()
        
        print("Starting real-time detection...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 将帧添加到处理队列
                self.frame_queue.put(frame.copy())  # 移除timeout限制
                
                # 显示带结果的帧
                processed_frame = self.display_frame(frame)
                cv2.imshow('Real-time Fall Detection', processed_frame)
                out.write(processed_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # 设置停止标志并等待处理线程结束
            self.stop_flag.set()
            process_thread.join()
            
            # 释放资源
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"Detection finished! Video saved to: {output_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Real-time Fall Detection')
    
    # 输入输出参数
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera ID (default: 0)')
    parser.add_argument('--output_dir', type=str, default='./demo/output_detect/',
                      help='Output directory')
    parser.add_argument('--model_dir', type=str, default='checkpoint/fall_detection_lstm/2025-03-22-2328-b',
                      help='Fall detection model directory')
    
    # 硬件参数
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU ID')
    
    # 姿态估计参数
    parser.add_argument('--fix_z', action='store_true',
                      help='Fix Z axis')
    parser.add_argument('--detector', type=str, default='yolo11',
                      help='Human detector type')
    parser.add_argument('--batch_size', type=int, default=2000,
                      help='Frame batch size (default: 2000)')
    
    # 实时处理参数
    parser.add_argument('--buffer_size', type=int, default=120,
                       help='Frame buffer size (default: 120)')
    parser.add_argument('--overlap_ratio', type=float, default=0.1,
                       help='Overlap ratio between consecutive frame batches (0-1, default: 0.1)')
    parser.add_argument('--max_pending_frames', type=int, default=200,
                       help='Maximum pending frames before adaptive sampling (default: 100)')
    parser.add_argument('--sampling_rate', type=float, default=0.5,
                       help='Sampling rate when max_pending_frames is exceeded (default: 0.5)')
    
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 创建实时跌倒检测器并启动摄像头检测
    detector = RealtimeFallDetection(args)
    detector.start_camera()

if __name__ == "__main__":
    main()