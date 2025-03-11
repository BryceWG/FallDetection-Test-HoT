# python render_3d_poses.py --npz demo\output\sample_video\output_3D\output_keypoints_3d.npz
import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import shutil  # 导入shutil模块用于文件操作
import multiprocessing as mp
from functools import partial
import psutil  # 用于获取系统内存信息

# 导入自定义模块
from utils import show3Dpose, showimage
from data_processor import camera_to_world

def preprocess_pose_data(pose_data):
    """
    预处理姿态数据，提前进行坐标转换
    """
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
    processed_data = np.zeros_like(pose_data)
    
    for i in range(pose_data.shape[0]):
        frame_pose = pose_data[i].copy()
        frame_pose = camera_to_world(frame_pose, R=rot, t=0)
        frame_pose[:, 2] -= np.min(frame_pose[:, 2])
        processed_data[i] = frame_pose
    
    return processed_data

def render_single_frame(pose_data, frame_idx, output_path, fix_z=True, view_angles=None):
    """
    渲染单帧3D姿态数据
    
    参数:
        pose_data: 预处理后的3D姿态数据
        frame_idx: 要渲染的帧索引
        output_path: 输出图像路径
        fix_z: 是否固定z轴范围
        view_angles: 视角参数，格式为(elevation, azimuth)
    """
    # 获取指定帧的姿态数据（已预处理）
    frame_pose = pose_data[frame_idx]
    
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
    plt.title(f"Frame {frame_idx}", fontsize=14)
    
    # 保存图像
    plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight')
    plt.close('all')  # 确保关闭所有图形，释放内存
    
    return True

def get_optimal_process_count():
    """
    根据系统资源确定最优进程数
    """
    cpu_count = mp.cpu_count()
    memory = psutil.virtual_memory()
    
    # 根据可用内存和CPU核心数确定进程数
    # 每个进程预估使用500MB内存
    estimated_memory_per_process = 500 * 1024 * 1024  # 500MB in bytes
    max_processes_by_memory = int(memory.available * 0.7 / estimated_memory_per_process)
    
    # 取CPU核心数和内存限制的最小值，并确保至少有1个进程
    optimal_count = max(1, min(cpu_count - 1, max_processes_by_memory))
    return optimal_count

def process_frame(frame_info):
    """
    处理单个帧的辅助函数，用于多进程处理
    """
    pose_data, frame_idx, output_path, fix_z, view_angles = frame_info
    try:
        return render_single_frame(pose_data, frame_idx, output_path, fix_z, view_angles)
    except Exception as e:
        print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        return False

def render_sequence(pose_data, output_dir, fix_z=True, view_angles=None):
    """
    使用多进程渲染完整3D姿态数据序列
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 预处理姿态数据
    print("预处理姿态数据...")
    processed_data = preprocess_pose_data(pose_data)
    
    # 准备帧信息列表
    frame_infos = []
    for i in range(processed_data.shape[0]):
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame_infos.append((processed_data, i, output_path, fix_z, view_angles))
    
    # 获取最优进程数
    num_processes = get_optimal_process_count()
    print(f"使用 {num_processes} 个进程进行并行渲染...")
    
    # 分批处理以控制内存使用
    batch_size = 50  # 每批处理50帧
    for i in range(0, len(frame_infos), batch_size):
        batch = frame_infos[i:i + batch_size]
        with mp.Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_frame, batch), 
                     total=len(batch),
                     desc=f"渲染批次 {i//batch_size + 1}/{len(frame_infos)//batch_size + 1}"))
    
    print(f"已成功渲染 {processed_data.shape[0]} 帧到 {output_dir}")
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

def main():
    parser = argparse.ArgumentParser(description='3D姿态数据渲染工具')
    parser.add_argument('--npz', type=str, required=True, help='输入的.npz文件路径')
    parser.add_argument('--output_dir', type=str, default='./demo/render_output/', help='输出根目录')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴范围')
    parser.add_argument('--elev', type=float, default=15, help='视角仰角（度）')
    parser.add_argument('--azim', type=float, default=70, help='视角方位角（度）')
    
    args = parser.parse_args()
    
    # 加载.npz文件
    try:
        data = np.load(args.npz, allow_pickle=True)
        pose_data = data['reconstruction']
        
        # 处理不同的数据形状
        if len(pose_data.shape) == 4:  # [B, T, J, 3]
            pose_data = pose_data[0]  # 取第一个批次
        
        print(f"Loaded 3D pose data, shape: {pose_data.shape}")
    except Exception as e:
        print(f"加载.npz文件时出错: {e}")
        return
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取输入文件的基本名称（不含扩展名）并创建同名文件夹
    input_basename = os.path.splitext(os.path.basename(args.npz))[0]
    output_subdir = os.path.join(args.output_dir, input_basename)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 设置sequence目录路径
    sequence_dir = os.path.join(output_subdir, "sequence")
    
    # 如果sequence目录已存在，先清空它
    if os.path.exists(sequence_dir):
        print(f"清空目标文件夹: {sequence_dir}")
        shutil.rmtree(sequence_dir)
    
    # 创建新的sequence目录
    os.makedirs(sequence_dir, exist_ok=True)
    
    # 渲染序列并创建视频
    render_sequence(pose_data, sequence_dir, args.fix_z, (args.elev, args.azim))
    
    video_path = os.path.join(output_subdir, "3d_pose_video.mp4")
    create_video_from_frames(sequence_dir, video_path, args.fps)

if __name__ == "__main__":
    main() 