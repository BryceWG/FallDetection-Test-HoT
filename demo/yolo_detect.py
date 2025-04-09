import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys
from contextlib import contextmanager
from lib.yolo11.human_detector import load_model, yolo_human_det, get_default_args

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

def draw_bbox(image, bbox, score, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框和置信度分数
    """
    # 确保边界框坐标是整数
    bbox = [float(x) for x in bbox]  # 先转换为float以处理numpy数组
    x1, y1, x2, y2 = map(int, bbox)  # 然后转换为int
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # 添加置信度分数标签
    text = f'Person: {float(score):.2f}'  # 确保score也转换为float
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
    处理视频文件，对每一帧进行人体检测并保存结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_subdir = os.path.join(output_dir, video_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'错误: 无法打开视频文件 {video_path}')
        return False
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f'\n处理视频: {video_name}')
    print(f'总帧数: {total_frames}')
    print(f'分辨率: {frame_width}x{frame_height}')
    print(f'帧率: {fps}')
    
    # 逐帧处理
    frame_idx = 0
    detections_count = 0
    
    with tqdm(total=total_frames, desc='处理进度', ncols=100) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 使用YOLO11进行人体检测
            try:
                with suppress_stdout():
                    bboxs_batch, scores_batch = yolo_human_det([frame], model, 
                                                             reso=args.det_dim, 
                                                             confidence=args.confidence,
                                                             quiet=True)
                    
                # 处理检测结果
                if bboxs_batch[0] is not None:
                    result_frame = frame.copy()
                    for bbox, score in zip(bboxs_batch[0], scores_batch[0]):
                        result_frame = draw_bbox(result_frame, bbox, score)
                    detections_count += 1
                else:
                    result_frame = frame
                
                # 保存结果图片
                output_path = os.path.join(output_subdir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(output_path, result_frame)
                
            except Exception as e:
                print(f'\n处理第 {frame_idx} 帧时出错: {str(e)}')
                
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f'\n视频处理完成!')
    print(f'检测到人体的帧数: {detections_count}')
    print(f'检测结果保存在: {output_subdir}')
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频人体检测工具')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output_dir', type=str, default='output_frames', help='输出目录路径')
    parser.add_argument('--confidence', type=float, default=0.5, help='检测置信度阈值')
    
    args_cmd = parser.parse_args()
    
    # 获取YOLO默认参数并更新置信度
    model_args = get_default_args()
    model_args.confidence = args_cmd.confidence
    
    # 加载YOLO11模型
    print('加载YOLO11模型...')
    with suppress_stdout():
        model = load_model(model_args)
    
    # 处理视频
    process_video(args_cmd.video_path, args_cmd.output_dir, model, model_args)

if __name__ == '__main__':
    main()