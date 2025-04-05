import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
import sys
from contextlib import contextmanager
from lib.yolo11.human_detector import load_model, yolo_human_det, reset_target, get_default_args

@contextmanager
def suppress_stdout():
    """
    临时禁用标准输出
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, text=None):
    """
    在图像上绘制边界框和可选的文本标签
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if text:
        # 计算文本大小以确定背景矩形的尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 绘制文本背景矩形
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # 绘制文本
        cv2.putText(image, text, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return image

def process_video(video_path, output_dir, model, args):
    """
    处理单个视频文件，检测目标锁定情况
    持续检测直到发现人体
    """
    # 重置目标锁定状态
    reset_target()
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'status': 'error',
            'message': '无法打开视频文件',
            'frame_count': 0,
            'area': None,
            'lock_info': None
        }
    
    # 获取视频信息
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_count = 0
    batch_size = 3  # 每次处理3帧
    
    while frame_count < total_frames:
        # 读取一批帧
        batch_frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            frame_count += 1
        
        if not batch_frames:
            break
            
        # 使用YOLO11进行人体检测
        try:
            with suppress_stdout():
                bboxs_batch, scores_batch = yolo_human_det(batch_frames, model, reso=args.det_dim, confidence=args.confidence, quiet=True)
        except Exception as e:
            cap.release()
            return {
                'status': 'error',
                'message': f'检测过程出错: {str(e)}',
                'frame_count': frame_count,
                'area': None,
                'lock_info': None
            }
        
        # 检查是否检测到人体
        for i, bboxs in enumerate(bboxs_batch):
            if bboxs is not None:
                # 计算检测框面积
                x1, y1, x2, y2 = bboxs[0]  # 取第一个检测框
                area = (x2 - x1) * (y2 - y1)
                
                # 计算实际的帧号
                actual_frame_idx = frame_count - len(batch_frames) + i
                
                # 在检测到的帧上绘制检测框
                result_frame = batch_frames[i].copy()
                area_text = f'Frame: {actual_frame_idx}, Area: {area:.0f}px²'
                result_frame = draw_bbox(result_frame, bboxs[0], text=area_text)
                
                # 保存结果图片
                output_path = os.path.join(output_dir, f'{video_name}_detection_frame_{actual_frame_idx}.jpg')
                cv2.imwrite(output_path, result_frame)
                
                cap.release()
                return {
                    'status': 'success',
                    'message': '成功锁定目标',
                    'frame_count': actual_frame_idx,
                    'area': area,
                    'frame_idx': actual_frame_idx,
                    'lock_info': f'已锁定目标! 帧号: {actual_frame_idx}, 面积: {area:.2f}'
                }
    
    # 如果遍历完所有帧都没有检测到人体
    cap.release()
    return {
        'status': 'warning',
        'message': '未检测到任何人体',
        'frame_count': frame_count,
        'area': None,
        'lock_info': None
    }

def batch_process_videos(video_dir, output_dir):
    """
    批量处理视频文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext}')))
    
    if not video_files:
        print(f'错误: 在目录 {video_dir} 中未找到视频文件')
        return
    
    # 加载YOLO11模型
    with suppress_stdout():
        args = get_default_args()
        model = load_model(args)
    
    # 处理结果统计
    results = {
        'success': 0,
        'warning': 0,
        'error': 0,
        'details': []
    }
    
    print(f'开始处理 {len(video_files)} 个视频文件...')
    
    # 处理每个视频文件
    for video_path in tqdm(video_files, desc='处理进度', ncols=100, dynamic_ncols=False):
        video_name = os.path.basename(video_path)
        result = process_video(video_path, output_dir, model, args)
        
        # 更新统计信息
        results[result['status']] += 1
        results['details'].append({
            'video_name': video_name,
            'status': result['status'],
            'message': result['message'],
            'frame_count': result['frame_count'],
            'area': result['area'],
            'lock_info': result['lock_info']
        })
    
    # 生成报告
    report_path = os.path.join(output_dir, 'detection_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('人体检测目标锁定测试报告\n')
        f.write('=' * 50 + '\n\n')
        
        # 写入统计信息
        f.write('统计信息:\n')
        f.write(f'- 成功: {results["success"]}\n')
        f.write(f'- 警告: {results["warning"]}\n')
        f.write(f'- 错误: {results["error"]}\n\n')
        
        # 写入详细信息
        f.write('详细信息:\n')
        f.write('-' * 50 + '\n')
        for detail in results['details']:
            f.write(f'视频: {detail["video_name"]}\n')
            f.write(f'状态: {detail["status"]}\n')
            f.write(f'信息: {detail["message"]}\n')
            f.write(f'处理帧数: {detail["frame_count"]}\n')
            if detail['area'] is not None:
                f.write(f'目标面积: {detail["area"]:.0f}px²\n')
            if detail['lock_info'] is not None:
                f.write(f'锁定信息: {detail["lock_info"]}\n')
            f.write('-' * 50 + '\n')
    
    print('\n处理完成!')
    print(f'成功: {results["success"]}')
    print(f'警告: {results["warning"]}')
    print(f'错误: {results["error"]}')
    print(f'\n检测报告已保存至: {report_path}')
    print(f'检测结果图片已保存至: {output_dir}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='视频人体检测目标锁定测试工具')
    parser.add_argument('--video_dir', type=str, required=True, help='输入视频目录路径')
    parser.add_argument('--output_dir', type=str, default='detection_results', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 运行批处理
    batch_process_videos(args.video_dir, args.output_dir)
