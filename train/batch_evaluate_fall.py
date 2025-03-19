#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python train/batch_evaluate_fall.py --model_path checkpoint\fall_detection_lstm\2025-03-17-2321-b\best_model.pth --csv_file train/frame_data.csv --pose_dir demo/output --output_dir train/evaluation_results --use_training_params
import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from predict_fall import FallPredictor, load_training_params

class BatchFallEvaluator:
    """
    批量跌倒评估器
    用于评估多个视频的跌倒检测结果
    """
    def __init__(self, predictor, window_size=30, stride=15, threshold=0.5):
        """
        初始化评估器
        
        参数:
            predictor: FallPredictor实例
            window_size: 滑动窗口大小
            stride: 滑动步长
            threshold: 跌倒判定阈值
        """
        self.predictor = predictor
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        
    def evaluate_sequence(self, pose_data, true_start=None, true_end=None, has_fall=False):
        """
        评估单个序列
        
        参数:
            pose_data: 3D姿态数据
            true_start: 真实跌倒开始帧
            true_end: 真实跌倒结束帧
            has_fall: 是否包含跌倒
            
        返回:
            评估结果字典
        """
        # 获取预测结果
        predictions, fall_windows = self.predictor.predict(
            pose_data, 
            threshold=self.threshold,
            window_size=self.window_size,
            stride=self.stride
        )
        
        # 生成帧级别的标签和预测
        total_frames = len(pose_data)
        frame_labels = np.zeros(total_frames)
        frame_predictions = np.zeros(total_frames)
        
        # 设置真实标签
        if has_fall and true_start is not None and true_end is not None:
            # 确保索引是整数
            start_idx = int(true_start)
            end_idx = int(true_end)
            if 0 <= start_idx < total_frames and 0 <= end_idx < total_frames:
                frame_labels[start_idx:end_idx+1] = 1
            
        # 设置预测标签
        for pred in predictions:
            start_frame = int(pred['start_frame'])
            end_frame = int(pred['end_frame'])
            if 0 <= start_frame < total_frames and 0 <= end_frame < total_frames:
                is_fall = 1 if pred['probability'] >= self.threshold else 0
                frame_predictions[start_frame:end_frame] = is_fall
            
        return {
            'predictions': predictions,
            'fall_windows': fall_windows,
            'frame_labels': frame_labels,
            'frame_predictions': frame_predictions
        }
        
    def find_optimal_threshold(self, all_frame_labels, all_predictions, thresholds=None):
        """
        寻找最优的分类阈值
        
        参数:
            all_frame_labels: 真实标签数组
            all_predictions: 预测概率数组
            thresholds: 要评估的阈值列表，如果为None则自动生成
            
        返回:
            最优阈值和对应的评估结果
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
            
        best_threshold = 0.5
        best_accuracy = 0
        best_results = None
        
        print("\n开始寻找最优阈值...")
        for threshold in tqdm(thresholds, desc="评估阈值"):
            # 使用当前阈值进行分类
            binary_preds = (np.array(all_predictions) >= threshold).astype(int)
            
            # 计算性能指标
            classification_rep = classification_report(all_frame_labels, binary_preds, output_dict=True)
            conf_matrix = confusion_matrix(all_frame_labels, binary_preds)
            
            # 获取当前准确率
            current_accuracy = classification_rep['accuracy']
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_threshold = threshold
                best_results = {
                    'threshold': threshold,
                    'accuracy': current_accuracy,
                    'classification_report': classification_rep,
                    'confusion_matrix': conf_matrix
                }
                
        print(f"\n找到最优阈值: {best_threshold:.3f}")
        print(f"最佳准确率: {best_accuracy:.4f}")
        
        return best_results

    def evaluate_dataset(self, csv_file, base_pose_dir):
        """
        评估整个数据集
        
        参数:
            csv_file: 包含视频信息的CSV文件路径
            base_pose_dir: 3D姿态文件的基础目录
            
        返回:
            评估结果字典
        """
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        all_frame_labels = []
        all_frame_predictions = []
        all_frame_probabilities = []  # 存储原始预测概率
        results = []
        
        # 遍历每个视频
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理视频"):
            video_id = row['video_id']
            has_fall = row['has_fall']
            fall_start = row['fall_start_frame'] if has_fall else None
            fall_end = row['fall_end_frame'] if has_fall else None
            
            # 构建姿态文件路径
            pose_file = os.path.join(base_pose_dir, video_id, 'output_3D', 'output_keypoints_3d.npz')
            
            if not os.path.exists(pose_file):
                print(f"警告: 找不到姿态文件 {pose_file}")
                continue
                
            try:
                # 加载姿态数据
                pose_data = np.load(pose_file, allow_pickle=True)['reconstruction']
                
                # 评估序列
                eval_result = self.evaluate_sequence(
                    pose_data,
                    true_start=fall_start,
                    true_end=fall_end,
                    has_fall=has_fall
                )
                
                # 收集结果
                all_frame_labels.extend(eval_result['frame_labels'])
                all_frame_predictions.extend(eval_result['frame_predictions'])
                
                # 收集原始预测概率
                frame_probabilities = np.zeros(len(pose_data))
                for pred in eval_result['predictions']:
                    start_frame = int(pred['start_frame'])
                    end_frame = int(pred['end_frame'])
                    if 0 <= start_frame < len(pose_data) and 0 <= end_frame < len(pose_data):
                        frame_probabilities[start_frame:end_frame] = pred['probability']
                all_frame_probabilities.extend(frame_probabilities)
                
                # 添加视频级别的结果
                results.append({
                    'video_id': video_id,
                    'has_fall': has_fall,
                    'fall_start': fall_start,
                    'fall_end': fall_end,
                    'predictions': eval_result['predictions'],
                    'detected_falls': len(eval_result['fall_windows'])
                })
                
            except Exception as e:
                print(f"处理视频 {video_id} 时出错: {str(e)}")
                continue
        
        # 寻找最优阈值
        all_frame_labels = np.array(all_frame_labels)
        all_frame_probabilities = np.array(all_frame_probabilities)
        optimal_results = self.find_optimal_threshold(all_frame_labels, all_frame_probabilities)
        
        return {
            'video_results': results,
            'classification_report': optimal_results['classification_report'],
            'confusion_matrix': optimal_results['confusion_matrix'],
            'optimal_threshold': optimal_results['threshold']
        }
        
    def save_results(self, results, output_dir):
        """
        保存评估结果
        
        参数:
            results: evaluate_dataset返回的结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存整体评估结果
        summary_file = os.path.join(output_dir, 'evaluation_summary.json')
        summary = {
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'optimal_threshold': results['optimal_threshold']
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        # 保存每个视频的详细结果
        details_file = os.path.join(output_dir, 'video_details.csv')
        video_results = []
        
        for video_result in results['video_results']:
            # 统计预测的跌倒概率
            pred_probs = [pred['probability'] for pred in video_result['predictions']]
            max_prob = max(pred_probs) if pred_probs else 0
            avg_prob = np.mean(pred_probs) if pred_probs else 0
            
            video_results.append({
                'video_id': video_result['video_id'],
                'true_has_fall': video_result['has_fall'],
                'true_fall_start': video_result['fall_start'],
                'true_fall_end': video_result['fall_end'],
                'detected_falls': video_result['detected_falls'],
                'max_fall_probability': max_prob,
                'avg_fall_probability': avg_prob,
                'prediction_correct': (video_result['has_fall'] == (video_result['detected_falls'] > 0))
            })
            
        pd.DataFrame(video_results).to_csv(details_file, index=False)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        conf_matrix = results['confusion_matrix']
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        class_names = ['Normal', 'Fall']
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # 显示数值
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        print(f"\n评估结果已保存至: {output_dir}")
        print(f"最优分类阈值: {results['optimal_threshold']:.3f}")
        print(f"\n分类报告:")
        # 直接使用已经计算好的分类报告
        classification_rep = results['classification_report']
        
        # 找到正确的键名
        normal_key = [k for k in classification_rep.keys() if k.startswith('0')][0]
        fall_key = [k for k in classification_rep.keys() if k.startswith('1')][0]
        
        print("\n精确率:")
        print(f"Normal (0): {classification_rep[normal_key]['precision']:.4f}")
        print(f"Fall (1): {classification_rep[fall_key]['precision']:.4f}")
        print("\n召回率:")
        print(f"Normal (0): {classification_rep[normal_key]['recall']:.4f}")
        print(f"Fall (1): {classification_rep[fall_key]['recall']:.4f}")
        print("\nF1分数:")
        print(f"Normal (0): {classification_rep[normal_key]['f1-score']:.4f}")
        print(f"Fall (1): {classification_rep[fall_key]['f1-score']:.4f}")
        print(f"\n整体准确率: {classification_rep['accuracy']:.4f}")

def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='批量评估跌倒检测模型')
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型checkpoint路径')
    parser.add_argument('--csv_file', type=str, required=True,
                      help='包含视频信息的CSV文件路径')
    parser.add_argument('--pose_dir', type=str, required=True,
                      help='3D姿态文件的基础目录')
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
    parser.add_argument('--output_dir', type=str, default='./batch_evaluation_results',
                      help='结果保存目录')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU ID')
    parser.add_argument('--use_training_params', action='store_true',
                      help='是否使用training_summary.json中的训练参数')
    
    args = parser.parse_args()
    
    # 如果指定了使用训练参数,则尝试加载
    if args.use_training_params:
        training_params = load_training_params(args.model_path)
        if training_params and 'parameters' in training_params:
            params = training_params['parameters']
            
            # 更新模型参数
            if 'model_params' in params:
                model_params = params['model_params']
                args.hidden_dim = model_params.get('hidden_dim', args.hidden_dim)
                args.num_layers = model_params.get('num_layers', args.num_layers)
                args.dropout = model_params.get('dropout', args.dropout)
                print("\n使用训练时的模型参数:")
                print(f"hidden_dim: {args.hidden_dim}")
                print(f"num_layers: {args.num_layers}")
                print(f"dropout: {args.dropout}")
            
            # 更新数据处理参数
            if 'data_params' in params:
                data_params = params['data_params']
                args.window_size = data_params.get('fall_seq_length', args.window_size)
                args.stride = data_params.get('fall_stride', args.stride)
                print("\n使用训练时的数据处理参数:")
                print(f"window_size: {args.window_size}")
                print(f"stride: {args.stride}")
    
    return args

def main():
    """
    主函数
    """
    args = process_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 创建预测器
        predictor = FallPredictor(
            model_path=args.model_path,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device
        )
        
        # 创建评估器
        evaluator = BatchFallEvaluator(
            predictor=predictor,
            window_size=args.window_size,
            stride=args.stride,
            threshold=args.threshold
        )
        
        # 执行批量评估
        results = evaluator.evaluate_dataset(
            csv_file=args.csv_file,
            base_pose_dir=args.pose_dir
        )
        
        # 保存结果
        evaluator.save_results(results, args.output_dir)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()