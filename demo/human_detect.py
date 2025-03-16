from ultralytics import YOLO
import cv2
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='使用YOLO进行人体检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    parser.add_argument('--confidence', type=float, default=0.5, help='检测置信度阈值')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用的设备(cuda/cpu)')
    
    args = parser.parse_args()
    
    # 如果未指定输出路径,则在输入视频的同目录下生成
    if args.output is None:
        video_dir = os.path.dirname(args.video)
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(video_dir, f'{video_name}_detected.mp4')
        
    return args

def draw_boxes(frame, boxes, scores):
    """
    在帧上绘制检测框和置信度
    """
    for box, score in zip(boxes, scores):
        # 获取坐标
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制矩形框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加置信度标签
        label = f'Person {score:.2f}'
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video(args):
    """
    处理视频并进行人体检测
    """
    print('⏳ 初始化模型...')
    # 加载YOLO模型
    model = YOLO('D:\CodeSpace\FallDetection\FallDetection-Test-HoT\demo\lib\checkpoint\yolo11s.pt')  # 使用YOLOv8 nano模型
    
    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f'无法打开视频: {args.video}')
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f'✓ 视频信息:')
    print(f'   - 分辨率: {width}x{height}')
    print(f'   - 帧率: {fps}')
    print(f'   - 总帧数: {total_frames}')
    print(f'   - 输出文件: {args.output}')
    print('\n⏳ 开始处理...')
    
    try:
        # 使用tqdm创建进度条
        for _ in tqdm(range(total_frames), desc='处理进度'):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用YOLO进行检测
            results = model(frame, conf=args.confidence, classes=0)  # classes=0 表示只检测人
            
            if len(results) > 0:
                result = results[0]  # 获取第一帧的结果
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # 获取所有检测框和置信度
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    # 在帧上绘制检测结果
                    frame = draw_boxes(frame, boxes, scores)
            
            # 写入帧
            out.write(frame)
            
    except KeyboardInterrupt:
        print('\n⚠️ 处理被用户中断')
    finally:
        # 释放资源
        cap.release()
        out.release()
        
    print(f'\n✓ 处理完成! 输出已保存到: {args.output}')

def main():
    # 解析命令行参数
    args = get_args()
    
    try:
        # 处理视频
        process_video(args)
    except Exception as e:
        print(f'\n❌ 发生错误: {str(e)}')
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 