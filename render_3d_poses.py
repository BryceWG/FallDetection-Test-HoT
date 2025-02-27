import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# 导入自定义模块
from utils import show3Dpose, showimage
from data_processor import camera_to_world


def render_single_frame(pose_data, frame_idx, output_path, fix_z=True, view_angles=None):
    """
    渲染单帧3D姿态数据
    
    参数:
        pose_data: 3D姿态数据，形状为[T, 17, 3]
        frame_idx: 要渲染的帧索引
        output_path: 输出图像路径
        fix_z: 是否固定z轴范围
        view_angles: 视角参数，格式为(elevation, azimuth)
    """
    # 确保索引在有效范围内
    if frame_idx >= pose_data.shape[0]:
        print(f"错误：帧索引 {frame_idx} 超出范围，最大索引为 {pose_data.shape[0]-1}")
        return False
    
    # 获取指定帧的姿态数据
    frame_pose = pose_data[frame_idx].copy()
    
    # 坐标系转换
    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    frame_pose = camera_to_world(frame_pose, R=rot, t=0)
    
    # 调整z轴
    frame_pose[:, 2] -= np.min(frame_pose[:, 2])
    
    # 创建图形
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    
    # 设置视角（如果提供）
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # 显示3D姿态
    show3Dpose(frame_pose, ax, fix_z)
    
    # 添加帧索引标题
    plt.title(f"帧 {frame_idx}", fontsize=14)
    
    # 保存图像
    plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight')
    plt.close()
    
    return True


def render_sequence(pose_data, output_dir, fix_z=True, start_frame=0, end_frame=None, step=1, view_angles=None):
    """
    渲染一系列3D姿态数据帧
    
    参数:
        pose_data: 3D姿态数据，形状为[T, 17, 3]
        output_dir: 输出目录
        fix_z: 是否固定z轴范围
        start_frame: 起始帧索引
        end_frame: 结束帧索引（如果为None，则使用最后一帧）
        step: 帧间隔
        view_angles: 视角参数，格式为(elevation, azimuth)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定结束帧
    if end_frame is None:
        end_frame = pose_data.shape[0]
    else:
        end_frame = min(end_frame, pose_data.shape[0])
    
    # 渲染指定范围的帧
    print(f"渲染3D姿态序列（帧 {start_frame} 到 {end_frame-1}，步长 {step}）...")
    for i in tqdm(range(start_frame, end_frame, step)):
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        render_single_frame(pose_data, i, output_path, fix_z, view_angles)
    
    print(f"已成功渲染 {(end_frame-start_frame)//step} 帧到 {output_dir}")
    
    return True


def create_video_from_frames(frames_dir, output_video_path, fps=30):
    """
    从一系列图像帧创建视频
    
    参数:
        frames_dir: 包含图像帧的目录
        output_video_path: 输出视频路径
        fps: 视频帧率
    """
    # 获取所有图像帧
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    
    if not frame_files:
        print(f"错误：在 {frames_dir} 中未找到图像帧")
        return False
    
    # 读取第一帧以获取尺寸
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入所有帧
    print(f"创建视频 {output_video_path}...")
    for frame_file in tqdm(frame_files):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    
    # 释放资源
    video_writer.release()
    
    print(f"视频已成功创建：{output_video_path}")
    return True


def render_multi_view(pose_data, frame_idx, output_dir, fix_z=True):
    """
    从多个视角渲染单帧3D姿态
    
    参数:
        pose_data: 3D姿态数据，形状为[T, 17, 3]
        frame_idx: 要渲染的帧索引
        output_dir: 输出目录
        fix_z: 是否固定z轴范围
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义不同的视角
    views = [
        (15, 70, "正面视图"),    # 默认视图
        (15, 0, "侧面视图_右"),   # 右侧视图
        (15, 180, "侧面视图_左"),  # 左侧视图
        (90, 0, "俯视图"),       # 俯视图
        (0, 70, "水平视图")      # 水平视图
    ]
    
    # 渲染每个视角
    for elev, azim, name in views:
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_{name}.png")
        
        # 创建图形
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 获取指定帧的姿态数据
        frame_pose = pose_data[frame_idx].copy()
        
        # 坐标系转换
        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        frame_pose = camera_to_world(frame_pose, R=rot, t=0)
        
        # 调整z轴
        frame_pose[:, 2] -= np.min(frame_pose[:, 2])
        
        # 显示3D姿态
        show3Dpose(frame_pose, ax, fix_z)
        
        # 添加视角标题
        plt.title(f"{name} - 帧 {frame_idx}", fontsize=14)
        
        # 保存图像
        plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight')
        plt.close()
    
    # 创建组合视图
    create_combined_view(output_dir, frame_idx, views)
    
    print(f"已成功渲染帧 {frame_idx} 的多视角视图到 {output_dir}")
    return True


def create_combined_view(output_dir, frame_idx, views):
    """
    创建组合多视角视图
    
    参数:
        output_dir: 输出目录
        frame_idx: 帧索引
        views: 视角列表
    """
    # 读取所有视角图像
    images = []
    for _, _, name in views:
        img_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_{name}.png")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    if not images:
        return False
    
    # 确保所有图像具有相同尺寸
    height = min(img.shape[0] for img in images)
    width = min(img.shape[1] for img in images)
    
    resized_images = [cv2.resize(img, (width, height)) for img in images]
    
    # 创建网格布局
    rows = (len(images) + 2) // 3  # 每行最多3个图像
    cols = min(3, len(images))
    
    # 创建空白画布
    combined = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    combined.fill(255)  # 白色背景
    
    # 填充图像
    for i, img in enumerate(resized_images):
        r, c = i // cols, i % cols
        combined[r*height:(r+1)*height, c*width:(c+1)*width] = img
    
    # 保存组合图像
    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_combined.png")
    cv2.imwrite(output_path, combined)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='3D姿态数据渲染工具')
    parser.add_argument('--npz_file', type=str, required=True, help='输入的.npz文件路径')
    parser.add_argument('--output_dir', type=str, default='./demo/render_output/', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['single', 'sequence', 'video', 'multi_view'], 
                        default='single', help='渲染模式：单帧/序列/视频/多视角')
    parser.add_argument('--frame', type=int, default=0, help='要渲染的帧索引（用于单帧和多视角模式）')
    parser.add_argument('--start_frame', type=int, default=0, help='起始帧索引（用于序列和视频模式）')
    parser.add_argument('--end_frame', type=int, default=None, help='结束帧索引（用于序列和视频模式）')
    parser.add_argument('--step', type=int, default=1, help='帧间隔（用于序列和视频模式）')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率（用于视频模式）')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴范围')
    parser.add_argument('--elev', type=float, default=15, help='视角仰角（度）')
    parser.add_argument('--azim', type=float, default=70, help='视角方位角（度）')
    
    args = parser.parse_args()
    
    # 加载.npz文件
    try:
        data = np.load(args.npz_file, allow_pickle=True)
        pose_data = data['reconstruction']
        
        # 处理不同的数据形状
        if len(pose_data.shape) == 4:  # [B, T, J, 3]
            pose_data = pose_data[0]  # 取第一个批次
        
        print(f"已加载3D姿态数据，形状: {pose_data.shape}")
    except Exception as e:
        print(f"加载.npz文件时出错: {e}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式执行不同的渲染
    if args.mode == 'single':
        # 单帧渲染
        output_path = os.path.join(args.output_dir, f"frame_{args.frame:04d}.png")
        render_single_frame(pose_data, args.frame, output_path, args.fix_z, (args.elev, args.azim))
        print(f"已渲染帧 {args.frame} 到 {output_path}")
    
    elif args.mode == 'sequence':
        # 序列渲染
        sequence_dir = os.path.join(args.output_dir, "sequence")
        render_sequence(pose_data, sequence_dir, args.fix_z, args.start_frame, args.end_frame, args.step, (args.elev, args.azim))
    
    elif args.mode == 'video':
        # 视频渲染
        sequence_dir = os.path.join(args.output_dir, "sequence")
        render_sequence(pose_data, sequence_dir, args.fix_z, args.start_frame, args.end_frame, args.step, (args.elev, args.azim))
        
        video_path = os.path.join(args.output_dir, "3d_pose_video.mp4")
        create_video_from_frames(sequence_dir, video_path, args.fps)
    
    elif args.mode == 'multi_view':
        # 多视角渲染
        multi_view_dir = os.path.join(args.output_dir, "multi_view")
        render_multi_view(pose_data, args.frame, multi_view_dir, args.fix_z)


if __name__ == "__main__":
    main() 