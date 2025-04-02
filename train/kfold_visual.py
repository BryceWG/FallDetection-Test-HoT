# uv run train/kfold_visual.py --input_json checkpoint\fall_detection_lstm_3d\2025-04-01-0929\cv_results.json --output_dir checkpoint\fall_detection_lstm_3d\2025-04-01-0929
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def calculate_cv_metrics(fold_metrics):
    """
    计算交叉验证的平均指标
    """
    # 提取各折的测试指标
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    for fold_data in fold_metrics:
        test_results = fold_data['test_results']
        test_losses.append(test_results['test_loss'])
        
        # 从分类报告中提取指标
        class_report = test_results['classification_report']
        test_accuracies.append(class_report['accuracy'])
        
        # 提取跌倒类(类别1)的指标
        test_precisions.append(class_report['1']['precision'])
        test_recalls.append(class_report['1']['recall'])
        test_f1_scores.append(class_report['1']['f1-score'])
    
    # 计算平均指标
    avg_metrics = {
        'avg_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'avg_accuracy': np.mean(test_accuracies),
        'std_accuracy': np.std(test_accuracies),
        'avg_precision': np.mean(test_precisions),
        'std_precision': np.std(test_precisions),
        'avg_recall': np.mean(test_recalls),
        'std_recall': np.std(test_recalls),
        'avg_f1': np.mean(test_f1_scores),
        'std_f1': np.std(test_f1_scores)
    }
    
    # 打印平均指标
    print("\n交叉验证平均指标:")
    print(f"测试损失: {avg_metrics['avg_test_loss']:.4f} ± {avg_metrics['std_test_loss']:.4f}")
    print(f"准确率: {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    print(f"精确率: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"召回率: {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")
    print(f"F1分数: {avg_metrics['avg_f1']:.4f} ± {avg_metrics['std_f1']:.4f}")
    
    return avg_metrics


def plot_cv_metrics(fold_metrics, avg_metrics, output_dir):
    """
    绘制交叉验证指标图表
    """
    # 提取各折的测试准确率、精确率、召回率和F1分数
    folds = [data['fold'] for data in fold_metrics]
    accuracies = [data['test_results']['classification_report']['accuracy'] for data in fold_metrics]
    precisions = [data['test_results']['classification_report']['1']['precision'] for data in fold_metrics]
    recalls = [data['test_results']['classification_report']['1']['recall'] for data in fold_metrics]
    f1_scores = [data['test_results']['classification_report']['1']['f1-score'] for data in fold_metrics]
    
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    
    plt.plot(folds, accuracies, 'o-', label='Accuracy')
    plt.plot(folds, precisions, 's-', label='Precision')
    plt.plot(folds, recalls, '^-', label='Recall')
    plt.plot(folds, f1_scores, 'd-', label='F1 Score')
    
    # 添加平均值水平线
    plt.axhline(y=avg_metrics['avg_accuracy'], color='blue', linestyle='--', alpha=0.5, label='Mean Accuracy')
    plt.axhline(y=avg_metrics['avg_precision'], color='orange', linestyle='--', alpha=0.5, label='Mean Precision')
    plt.axhline(y=avg_metrics['avg_recall'], color='green', linestyle='--', alpha=0.5, label='Mean Recall')
    plt.axhline(y=avg_metrics['avg_f1'], color='red', linestyle='--', alpha=0.5, label='Mean F1')
    
    plt.xlabel('Fold Number')
    plt.ylabel('Metric Value')
    plt.title('Cross Validation Metrics by Fold')
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics.png'))
    print(f"图表已保存至: {os.path.join(output_dir, 'cv_metrics.png')}")
    plt.close()


def visualize_cv_results(input_json, output_dir):
    """
    从JSON文件加载交叉验证结果并进行可视化
    
    参数:
        input_json: 输入的JSON文件路径
        output_dir: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载JSON文件
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            cv_results = json.load(f)
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return
    
    # 提取数据
    fold_metrics = cv_results.get('fold_metrics')
    
    if not fold_metrics:
        print("JSON文件中没有找到fold_metrics数据")
        return
    
    # 计算平均指标
    if 'avg_metrics' in cv_results:
        avg_metrics = cv_results['avg_metrics']
        print("使用JSON文件中已有的平均指标")
    else:
        print("从折数据重新计算平均指标")
        avg_metrics = calculate_cv_metrics(fold_metrics)
    
    # 绘制图表
    plot_cv_metrics(fold_metrics, avg_metrics, output_dir)
    print(f"可视化完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化k折交叉验证结果')
    parser.add_argument('--input_json', type=str, required=True, help='输入的cv_results.json文件路径')
    parser.add_argument('--output_dir', type=str, default='cv_visualizations', help='输出文件夹路径')
    args = parser.parse_args()
    
    visualize_cv_results(args.input_json, args.output_dir)
