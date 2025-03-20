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
    def __init__(self, predictor, normal_seq_length=30, normal_stride=40,
                 fall_seq_length=30, fall_stride=10, overlap_threshold=0.35, threshold=0.5):
        """
        初始化评估器
        
        参数:
            predictor: FallPredictor实例
            normal_seq_length: 正常序列长度
            normal_stride: 正常序列步长
            fall_seq_length: 跌倒序列长度
            fall_stride: 跌倒序列步长
            overlap_threshold: 重叠阈值
            threshold: 跌倒判定阈值
        """
        self.predictor = predictor
        self.normal_seq_length = normal_seq_length
        self.normal_stride = normal_stride
        self.fall_seq_length = fall_seq_length
        self.fall_stride = fall_stride
        self.overlap_threshold = overlap_threshold
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
        sequence_results = []
        sequence_labels = []
        
        # 根据是否为跌倒序列选择不同的窗口参数
        seq_length = self.fall_seq_length if has_fall else self.normal_seq_length
        stride = self.fall_stride if has_fall else self.normal_stride
        
        # 遍历视频帧,生成序列
        total_frames = len(pose_data)
        for start_idx in range(0, total_frames - seq_length + 1, stride):
            end_idx = start_idx + seq_length
            
            # 如果剩余帧数不足一个完整序列,则跳过
            if end_idx > total_frames:
                break
                
            # 判断当前序列是否为跌倒序列
            if has_fall and true_start is not None and true_end is not None:
                # 计算序列与跌倒区间的重叠比例
                overlap_start = max(start_idx, true_start)
                overlap_end = min(end_idx, true_end)
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / seq_length
                    is_fall_seq = overlap_ratio >= self.overlap_threshold  # 使用与训练时相同的阈值
                else:
                    is_fall_seq = False
            else:
                is_fall_seq = False
                
            # 获取当前序列的预测结果
            sequence_data = pose_data[start_idx:end_idx]
            pred_prob = self.predictor.predict_sequence(sequence_data)
            
            sequence_results.append({
                'start_frame': start_idx,
                'end_frame': end_idx,
                'probability': float(pred_prob),
                'is_fall': int(pred_prob >= self.threshold)
            })
            sequence_labels.append(int(is_fall_seq))
            
        return {
            'predictions': sequence_results,
            'labels': sequence_labels,
            'fall_windows': [pred for pred in sequence_results if pred['is_fall']]
        }
        
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
        
        video_results = []
        all_video_labels = []
        all_video_predictions = []
        all_sequence_labels = []
        all_sequence_predictions = []
        all_sequence_probabilities = []
        
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
                
                # 收集序列级别的结果
                sequence_predictions = [pred['is_fall'] for pred in eval_result['predictions']]
                sequence_probabilities = [pred['probability'] for pred in eval_result['predictions']]
                sequence_labels = eval_result['labels']
                
                all_sequence_labels.extend(sequence_labels)
                all_sequence_predictions.extend(sequence_predictions)
                all_sequence_probabilities.extend(sequence_probabilities)
                
                # 收集视频级别的结果
                max_prob = max(pred['probability'] for pred in eval_result['predictions']) if eval_result['predictions'] else 0
                video_prediction = int(max_prob >= self.threshold)
                
                all_video_labels.append(has_fall)
                all_video_predictions.append(video_prediction)
                
                # 添加视频级别的结果
                video_results.append({
                    'video_id': video_id,
                    'has_fall': has_fall,
                    'fall_start': fall_start,
                    'fall_end': fall_end,
                    'predictions': eval_result['predictions'],
                    'sequence_labels': sequence_labels,
                    'detected_falls': len(eval_result['fall_windows']),
                    'max_probability': max_prob
                })
                
            except Exception as e:
                print(f"处理视频 {video_id} 时出错: {str(e)}")
                continue
        
        # 计算序列级别的性能指标
        sequence_classification_rep = classification_report(
            all_sequence_labels, 
            all_sequence_predictions, 
            output_dict=True,
            zero_division=0
        )
        sequence_conf_matrix = confusion_matrix(all_sequence_labels, all_sequence_predictions)
        
        # 寻找最优阈值
        optimal_results = self.find_optimal_threshold(
            all_sequence_labels,
            all_sequence_probabilities,
            level='sequence'
        )
        
        # 计算视频级别的性能指标
        video_classification_rep = classification_report(
            all_video_labels, 
            all_video_predictions, 
            output_dict=True,
            zero_division=0
        )
        video_conf_matrix = confusion_matrix(all_video_labels, all_video_predictions)
        
        return {
            'video_results': video_results,
            'video_classification_report': video_classification_rep,
            'video_confusion_matrix': video_conf_matrix,
            'sequence_classification_report': sequence_classification_rep,
            'sequence_confusion_matrix': sequence_conf_matrix,
            'optimal_threshold_results': optimal_results
        }
        
    def find_optimal_threshold(self, labels, probabilities, level='sequence', thresholds=None):
        """
        寻找最优分类阈值
        
        参数:
            labels: 真实标签列表
            probabilities: 预测概率列表
            level: 优化级别,可选'sequence'或'video'
            thresholds: 要评估的阈值列表，如果为None则自动生成
            
        返回:
            最优阈值和对应的评估结果
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.01)  # 从0.05到0.95，步长0.01
            
        best_threshold = 0.5
        best_accuracy = 0
        best_results = None
        
        print(f"\n开始寻找{level}级别最优阈值...")
        
        for threshold in tqdm(thresholds, desc="评估阈值"):
            # 使用当前阈值进行分类
            predictions = [int(prob >= threshold) for prob in probabilities]
            
            # 计算性能指标
            classification_rep = classification_report(labels, predictions, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(labels, predictions)
            
            # 使用整体准确率作为优化目标
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
                
        print("\n找到最优阈值: {:.3f}".format(best_threshold))
        print("最佳准确率: {:.4f}".format(best_accuracy))
        
        return best_results
        
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
            'video_level_results': {
                'classification_report': results['video_classification_report'],
                'confusion_matrix': results['video_confusion_matrix'].tolist(),
            },
            'sequence_level_results': {
                'classification_report': results['sequence_classification_report'],
                'confusion_matrix': results['sequence_confusion_matrix'].tolist(),
            },
            'optimal_threshold': {
                'value': results['optimal_threshold_results']['threshold'],
                'accuracy': results['optimal_threshold_results']['accuracy'],
                'classification_report': results['optimal_threshold_results']['classification_report'],
                'confusion_matrix': results['optimal_threshold_results']['confusion_matrix'].tolist()
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        # 保存每个视频的详细结果
        details_file = os.path.join(output_dir, 'video_details.csv')
        video_results = []
        
        for video_result in results['video_results']:
            video_results.append({
                'video_id': video_result['video_id'],
                'true_has_fall': video_result['has_fall'],
                'true_fall_start': video_result['fall_start'],
                'true_fall_end': video_result['fall_end'],
                'detected_falls': video_result['detected_falls'],
                'max_probability': video_result['max_probability'],
                'prediction_correct': (video_result['has_fall'] == (video_result['detected_falls'] > 0))
            })
            
        pd.DataFrame(video_results).to_csv(details_file, index=False)
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(
            results['video_confusion_matrix'], 
            'Video Level Confusion Matrix',
            os.path.join(output_dir, 'video_confusion_matrix.png')
        )
        
        self._plot_confusion_matrix(
            results['sequence_confusion_matrix'], 
            'Sequence Level Confusion Matrix',
            os.path.join(output_dir, 'sequence_confusion_matrix.png')
        )
        
        self._plot_confusion_matrix(
            results['optimal_threshold_results']['confusion_matrix'], 
            'Optimal Threshold Confusion Matrix',
            os.path.join(output_dir, 'optimal_threshold_confusion_matrix.png')
        )
        
        print("\n评估结果已保存至: {}".format(output_dir))
        
        print("\n序列级别评估结果:")
        self._print_metrics(results['sequence_classification_report'])
        
        print("\n视频级别评估结果:")
        self._print_metrics(results['video_classification_report'])
        
        print("\n最优阈值 ({:.3f}) 的评估结果:".format(results['optimal_threshold_results']['threshold']))
        self._print_metrics(results['optimal_threshold_results']['classification_report'])
        
    def _plot_confusion_matrix(self, conf_matrix, title, save_path):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
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
        plt.savefig(save_path)
        plt.close()
        
    def _print_metrics(self, classification_rep):
        """打印评估指标"""
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

def evaluate_multiple_models(base_model_dir, csv_file, pose_dir, output_dir, device, **kwargs):
    """
    评估指定目录下所有模型的性能
    
    参数:
        base_model_dir: 包含多个模型checkpoint的基础目录
        csv_file: 包含视频信息的CSV文件路径
        pose_dir: 3D姿态文件的基础目录
        output_dir: 结果保存目录
        device: 运行设备
        **kwargs: 其他参数,将传递给BatchFallEvaluator
    
    返回:
        包含所有模型评估结果的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有模型的结果
    all_results = {
        'sequence_level': [],
        'video_level': [],
        'optimal_threshold': []
    }
    
    # 遍历所有子目录查找模型文件
    for root, dirs, files in os.walk(base_model_dir):
        for file in files:
            if file == 'best_model.pth':
                model_path = os.path.join(root, file)
                print(f"\n正在评估模型: {model_path}")
                
                try:
                    # 尝试加载训练参数
                    training_summary_path = os.path.join(os.path.dirname(model_path), 'training_summary.json')
                    model_params = {}
                    data_params = {}
                    
                    if os.path.exists(training_summary_path):
                        try:
                            training_params = load_training_params(training_summary_path)
                            if training_params and 'parameters' in training_params:
                                params = training_params['parameters']
                                if 'model_params' in params:
                                    model_params = params['model_params']
                                if 'data_params' in params:
                                    data_params = params['data_params']
                        except Exception as e:
                            print(f"加载训练参数文件失败: {str(e)}, 跳过该模型")
                            continue
                    
                    # 创建预测器
                    try:
                        predictor = FallPredictor(
                            model_path=model_path,
                            hidden_dim=model_params.get('hidden_dim', kwargs.get('hidden_dim', 256)),
                            num_layers=model_params.get('num_layers', kwargs.get('num_layers', 3)),
                            dropout=model_params.get('dropout', kwargs.get('dropout', 0.3)),
                            device=device
                        )
                    except Exception as e:
                        print(f"加载模型失败: {str(e)}, 跳过该模型")
                        continue
                    
                    # 创建评估器
                    evaluator = BatchFallEvaluator(
                        predictor=predictor,
                        normal_seq_length=data_params.get('normal_seq_length', kwargs.get('normal_seq_length', 30)),
                        normal_stride=data_params.get('normal_stride', kwargs.get('normal_stride', 40)),
                        fall_seq_length=data_params.get('fall_seq_length', kwargs.get('fall_seq_length', 30)),
                        fall_stride=data_params.get('fall_stride', kwargs.get('fall_stride', 10)),
                        overlap_threshold=data_params.get('overlap_threshold', kwargs.get('overlap_threshold', 0.35)),
                        threshold=kwargs.get('threshold', 0.5)
                    )
                    
                    # 执行评估
                    results = evaluator.evaluate_dataset(csv_file, pose_dir)
                    
                    # 保存当前模型的结果
                    model_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(root)))
                    evaluator.save_results(results, model_output_dir)
                    
                    # 收集结果
                    model_result = {
                        'model_path': model_path,
                        'sequence_accuracy': results['sequence_classification_report']['accuracy'],
                        'video_accuracy': results['video_classification_report']['accuracy'],
                        'optimal_threshold': results['optimal_threshold_results']['threshold'],
                        'optimal_accuracy': results['optimal_threshold_results']['accuracy']
                    }
                    
                    all_results['sequence_level'].append(model_result.copy())
                    all_results['video_level'].append(model_result.copy())
                    all_results['optimal_threshold'].append(model_result.copy())
                    
                except Exception as e:
                    print(f"评估模型 {model_path} 时出错: {str(e)}")
                    continue
    
    # 对结果进行排序
    for key in all_results:
        all_results[key].sort(key=lambda x: x[key.replace('_level', '_accuracy') if key != 'optimal_threshold' else 'optimal_accuracy'], reverse=True)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # 打印最佳结果
    print("\n=== 评估结果汇总 ===")
    print("\n序列级别最佳模型:")
    best_sequence = all_results['sequence_level'][0]
    print(f"模型路径: {best_sequence['model_path']}")
    print(f"准确率: {best_sequence['sequence_accuracy']:.4f}")
    
    print("\n视频级别最佳模型:")
    best_video = all_results['video_level'][0]
    print(f"模型路径: {best_video['model_path']}")
    print(f"准确率: {best_video['video_accuracy']:.4f}")
    
    print("\n最优阈值最佳模型:")
    best_threshold = all_results['optimal_threshold'][0]
    print(f"模型路径: {best_threshold['model_path']}")
    print(f"准确率: {best_threshold['optimal_accuracy']:.4f}")
    print(f"最优阈值: {best_threshold['optimal_threshold']:.4f}")
    
    return all_results

def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='批量评估跌倒检测模型')
    parser.add_argument('--model_path', type=str, help='训练好的模型checkpoint路径')
    parser.add_argument('--model_dir', type=str, help='包含多个模型checkpoint的目录')
    parser.add_argument('--csv_file', type=str, required=True,
                      help='包含视频信息的CSV文件路径')
    parser.add_argument('--pose_dir', type=str, required=True,
                      help='3D姿态文件的基础目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='跌倒判定阈值')
    parser.add_argument('--normal_seq_length', type=int, default=30,
                      help='正常序列长度')
    parser.add_argument('--normal_stride', type=int, default=40,
                      help='正常序列步长')
    parser.add_argument('--fall_seq_length', type=int, default=30,
                      help='跌倒序列长度')
    parser.add_argument('--fall_stride', type=int, default=10,
                      help='跌倒序列步长')
    parser.add_argument('--overlap_threshold', type=float, default=0.35,
                      help='重叠阈值')
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
    
    if not args.model_path and not args.model_dir:
        parser.error("必须指定--model_path或--model_dir之一")
    
    if args.model_path and args.model_dir:
        parser.error("--model_path和--model_dir不能同时指定")
    
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
        if args.model_dir:
            # 评估多个模型
            evaluate_multiple_models(
                base_model_dir=args.model_dir,
                csv_file=args.csv_file,
                pose_dir=args.pose_dir,
                output_dir=args.output_dir,
                device=device,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                normal_seq_length=args.normal_seq_length,
                normal_stride=args.normal_stride,
                fall_seq_length=args.fall_seq_length,
                fall_stride=args.fall_stride,
                overlap_threshold=args.overlap_threshold,
                threshold=args.threshold
            )
        else:
            # 评估单个模型
            predictor = FallPredictor(
                model_path=args.model_path,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device
            )
            
            evaluator = BatchFallEvaluator(
                predictor=predictor,
                normal_seq_length=args.normal_seq_length,
                normal_stride=args.normal_stride,
                fall_seq_length=args.fall_seq_length,
                fall_stride=args.fall_stride,
                overlap_threshold=args.overlap_threshold,
                threshold=args.threshold
            )
            
            results = evaluator.evaluate_dataset(
                csv_file=args.csv_file,
                base_pose_dir=args.pose_dir
            )
            
            evaluator.save_results(results, args.output_dir)
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()