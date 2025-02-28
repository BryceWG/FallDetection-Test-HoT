import os
import cv2
import time
import copy
import torch
import argparse
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from queue import Queue
from collections import deque
from ultralytics import YOLO

# 导入自定义模块
from data_processor import normalize_screen_coordinates, convert_yolov11_to_hrnet_keypoints, camera_to_world
from utils import show2Dpose, show3Dpose

# 导入模型相关模块
import sys
sys.path.append(os.getcwd())
from model.mixste.hot_mixste import Model

# 设置matplotlib后端
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 全局变量
stop_signal = False
fps_stats = {'capture': 0, '2d_pose': 0, '3d_pose': 0, 'render': 0}
frame_counts = {'capture': 0, '2d_pose': 0, '3d_pose': 0, 'render': 0}
last_time = {'capture': time.time(), '2d_pose': time.time(), '3d_pose': time.time(), 'render': time.time()}

class FrameBuffer:
    """帧缓冲区，用于在不同处理阶段之间传递数据"""
    def __init__(self, maxsize=30):
        self.raw_frames = Queue(maxsize)
        self.pose2d_frames = Queue(maxsize)
        self.pose3d_data = Queue(maxsize)
        self.pose3d_frames = Queue(maxsize)
        self.display_frames = {'raw': None, 'pose2d': None, 'pose3d': None}
        self.lock = threading.Lock()
        
    def update_display_frame(self, frame_type, frame):
        with self.lock:
            self.display_frames[frame_type] = frame.copy()
            
    def get_display_frame(self, frame_type):
        with self.lock:
            if self.display_frames[frame_type] is None:
                return None
            return self.display_frames[frame_type].copy()


def update_fps(stage):
    """更新FPS统计信息"""
    global fps_stats, frame_counts, last_time
    current_time = time.time()
    frame_counts[stage] += 1
    
    # 每秒更新一次FPS
    if current_time - last_time[stage] >= 1.0:
        fps = frame_counts[stage] / (current_time - last_time[stage])
        fps_stats[stage] = round(fps, 1)
        frame_counts[stage] = 0
        last_time[stage] = current_time


def capture_thread(args, frame_buffer):
    """捕获视频帧的线程"""
    global stop_signal, fps_stats
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        stop_signal = True
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"摄像头分辨率: {width}x{height}")
    
    while not stop_signal:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            break
            
        # 如果缓冲区已满，移除最旧的帧
        if frame_buffer.raw_frames.full():
            try:
                frame_buffer.raw_frames.get_nowait()
            except:
                pass
                
        # 添加新帧到缓冲区
        try:
            frame_buffer.raw_frames.put_nowait(frame.copy())
        except:
            pass
            
        # 更新显示帧
        frame_buffer.update_display_frame('raw', frame)
        
        # 更新FPS
        update_fps('capture')
        
        # 控制捕获速率
        time.sleep(0.001)  # 小延迟以减少CPU使用
    
    cap.release()
    print("摄像头捕获线程已停止")


def pose2d_thread(args, frame_buffer):
    """2D姿态提取线程"""
    global stop_signal, fps_stats
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    
    # 上一帧的关键点和分数
    last_keypoints = None
    last_scores = None
    
    while not stop_signal:
        # 如果没有可用帧，等待
        if frame_buffer.raw_frames.empty():
            time.sleep(0.01)
            continue
            
        # 获取一帧进行处理
        try:
            frame = frame_buffer.raw_frames.get_nowait()
        except:
            continue
            
        # 使用YOLO进行姿态检测
        results = model(frame)
        
        # 检查是否检测到人体
        if len(results) > 0 and results[0].keypoints is not None:
            # 获取第一个检测到的人的关键点
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # [17, 3] 数组，每行是[x, y, conf]
            
            # 分离坐标和置信度
            kpts_xy = keypoints[:, :2]  # [17, 2]
            scores = keypoints[:, 2]  # [17]
            
            # 应用置信度阈值过滤
            if last_keypoints is not None:
                # 对于低置信度的关键点，使用上一帧的对应关键点
                low_conf_mask = scores < args.conf_thresh
                kpts_xy[low_conf_mask] = last_keypoints[low_conf_mask]
                scores[low_conf_mask] = last_scores[low_conf_mask]
            
            last_keypoints = kpts_xy.copy()
            last_scores = scores.copy()
        else:
            # 如果没有检测到人，使用上一帧的关键点或零填充
            if last_keypoints is not None:
                kpts_xy = last_keypoints.copy()
                scores = last_scores.copy()
            else:
                kpts_xy = np.zeros((17, 2))
                scores = np.zeros(17)
        
        # 将YOLOv11 Pose的关键点转换为HRNet格式
        keypoints_array = np.array([kpts_xy])[np.newaxis, ...]  # [1, 1, 17, 2]
        scores_array = np.array([scores])[np.newaxis, ...]  # [1, 1, 17]
        
        converted_keypoints = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
        converted_scores = scores_array.copy()  # 这里可以使用convert_yolov11_to_hrnet_scores函数
        
        # 绘制2D姿态
        pose2d_frame = frame.copy()
        pose2d_frame = show2Dpose(converted_keypoints[0, 0], pose2d_frame)
        
        # 添加关节点序号
        for i, kpt in enumerate(converted_keypoints[0, 0]):
            x, y = map(int, kpt)
            # 在关节点右下角显示序号，使用黑色背景增加可读性
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # 绘制黑色背景矩形
            cv2.rectangle(pose2d_frame, 
                         (x + 5, y + 5), 
                         (x + text_size[0] + 10, y + text_size[1] + 10),
                         (0, 0, 0), -1)
            
            # 绘制白色文字
            cv2.putText(pose2d_frame, text, (x + 8, y + text_size[1] + 7),
                       font, font_scale, (255, 255, 255), thickness)
        
        # 添加FPS信息
        cv2.putText(pose2d_frame, f"2D FPS: {fps_stats['2d_pose']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 如果3D姿态预测队列已满，移除最旧的帧
        if frame_buffer.pose2d_frames.full():
            try:
                frame_buffer.pose2d_frames.get_nowait()
            except:
                pass
                
        # 添加到2D姿态帧缓冲区
        try:
            frame_buffer.pose2d_frames.put_nowait({
                'frame': pose2d_frame.copy(),
                'keypoints': converted_keypoints.copy(),
                'scores': converted_scores.copy(),
                'img_size': frame.shape
            })
        except:
            pass
            
        # 更新显示帧
        frame_buffer.update_display_frame('pose2d', pose2d_frame)
        
        # 更新FPS
        update_fps('2d_pose')
        
        # 控制处理速率
        time.sleep(0.001)
    
    print("2D姿态提取线程已停止")


def pose3d_thread(args, frame_buffer):
    """3D姿态预测线程"""
    global stop_signal, fps_stats
    
    # 加载3D姿态预测模型
    print("加载3D姿态预测模型...")
    
    # 设置模型参数
    model_args = argparse.Namespace()
    model_args.layers, model_args.channel, model_args.d_hid, model_args.frames = 8, 512, 1024, 243
    model_args.token_num, model_args.layer_index = 81, 3
    model_args.pad = (model_args.frames - 1) // 2
    model_args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    model_args.n_joints, model_args.out_joints = 17, 17
    
    # 加载模型
    model = Model(model_args).cuda()
    
    # 加载预训练权重
    import glob
    model_path = sorted(glob.glob(os.path.join(model_args.previous_dir, '*.pth')))[0]
    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # 创建输入缓冲区
    input_buffer = deque(maxlen=model_args.frames)
    
    # 旋转参数
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
    
    while not stop_signal:
        # 如果没有可用的2D姿态帧，等待
        if frame_buffer.pose2d_frames.empty():
            time.sleep(0.01)
            continue
            
        # 获取一帧2D姿态数据
        try:
            pose2d_data = frame_buffer.pose2d_frames.get_nowait()
        except:
            continue
            
        # 提取数据
        keypoints = pose2d_data['keypoints']
        img_size = pose2d_data['img_size']
        
        # 归一化坐标
        input_2D = normalize_screen_coordinates(keypoints[0, 0], w=img_size[1], h=img_size[0])
        
        # 添加到输入缓冲区
        input_buffer.append(input_2D)
        
        # 如果缓冲区未满，继续收集帧
        if len(input_buffer) < model_args.frames:
            # 如果是第一帧，用它填充整个缓冲区
            if len(input_buffer) == 1:
                first_frame = input_buffer[0]
                for _ in range(model_args.frames - 1):
                    input_buffer.append(first_frame)
            continue
            
        # 准备模型输入
        input_2D_batch = np.array(list(input_buffer))
        
        # 数据增强（水平翻转）
        input_2D_aug = copy.deepcopy(input_2D_batch)
        input_2D_aug[:, :, 0] *= -1
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        
        # 准备最终输入
        input_2D_final = np.stack([input_2D_batch, input_2D_aug], axis=0)
        input_2D_final = input_2D_final[np.newaxis, :, :, :, :]
        input_2D_final = torch.from_numpy(input_2D_final.astype('float32')).cuda()
        
        # 预测3D姿态
        with torch.no_grad():
            output_3D_non_flip = model(input_2D_final[:, 0])
            output_3D_flip = model(input_2D_final[:, 1])
            
        # 处理翻转的输出
        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
        
        # 合并结果
        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        
        # 获取中间帧的预测结果
        mid_frame_idx = model_args.frames // 2
        output_3D[:, :, 0, :] = 0  # 设置根关节为原点
        pose3d = output_3D[0, mid_frame_idx].cpu().detach().numpy()
        
        # 坐标系转换
        pose3d_world = camera_to_world(pose3d, R=rot, t=0)
        pose3d_world[:, 2] -= np.min(pose3d_world[:, 2])  # 调整z轴
        
        # 如果3D姿态数据队列已满，移除最旧的数据
        if frame_buffer.pose3d_data.full():
            try:
                frame_buffer.pose3d_data.get_nowait()
            except:
                pass
                
        # 添加到3D姿态数据缓冲区
        try:
            frame_buffer.pose3d_data.put_nowait({
                'pose3d': pose3d_world.copy(),
                'frame': pose2d_data['frame'].copy()
            })
        except:
            pass
            
        # 更新FPS
        update_fps('3d_pose')
        
        # 控制处理速率
        time.sleep(0.001)
    
    print("3D姿态预测线程已停止")


def render_thread(args, frame_buffer):
    """3D姿态渲染线程"""
    global stop_signal, fps_stats
    
    while not stop_signal:
        # 如果没有可用的3D姿态数据，等待
        if frame_buffer.pose3d_data.empty():
            time.sleep(0.01)
            continue
            
        # 获取一帧3D姿态数据
        try:
            pose3d_data = frame_buffer.pose3d_data.get_nowait()
        except:
            continue
            
        # 提取数据
        pose3d = pose3d_data['pose3d']
        
        # 创建3D渲染
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 显示3D姿态
        show3Dpose(pose3d, ax, args.fix_z)
        
        # 添加FPS信息
        ax.set_title(f"3D FPS: {fps_stats['3d_pose']}", fontsize=12)
        
        # 渲染为图像
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # 转换为OpenCV格式
        pose3d_img = np.frombuffer(raw_data, dtype=np.uint8).reshape((size[1], size[0], 3))
        pose3d_img = cv2.cvtColor(pose3d_img, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        # 更新显示帧
        frame_buffer.update_display_frame('pose3d', pose3d_img)
        
        # 更新FPS
        update_fps('render')
        
        # 控制渲染速率
        time.sleep(0.001)
    
    print("3D姿态渲染线程已停止")


def display_thread(args, frame_buffer):
    """显示线程"""
    global stop_signal, fps_stats
    
    # 创建并配置显示窗口
    window_names = ["原始视频", "2D姿态", "3D姿态"]
    window_positions = [(50, 50), (700, 50), (50, 550)]
    window_size = (640, 480)
    
    # 只创建一次窗口
    for name, pos in zip(window_names, window_positions):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *window_size)
        cv2.moveWindow(name, *pos)
    
    while not stop_signal:
        # 获取显示帧
        raw_frame = frame_buffer.get_display_frame('raw')
        pose2d_frame = frame_buffer.get_display_frame('pose2d')
        pose3d_frame = frame_buffer.get_display_frame('pose3d')
        
        # 显示帧
        if raw_frame is not None:
            # 添加FPS信息
            display_frame = raw_frame.copy()
            cv2.putText(display_frame, f"摄像头 FPS: {fps_stats['capture']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_names[0], display_frame)
            
        if pose2d_frame is not None:
            cv2.imshow(window_names[1], pose2d_frame)
            
        if pose3d_frame is not None:
            cv2.imshow(window_names[2], pose3d_frame)
            
        # 检查退出键
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC或q键退出
            stop_signal = True
            break
            
        # 控制显示速率
        time.sleep(0.01)
    
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print("显示线程已停止")


def main():
    parser = argparse.ArgumentParser(description="在线3D姿态估计管道")
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--yolo_model', type=str, default='yolo11s-pose.pt', help='YOLO v11 Pose模型路径')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='关键点置信度阈值')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴范围')
    parser.add_argument('--buffer_size', type=int, default=30, help='帧缓冲区大小')
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 创建帧缓冲区
    frame_buffer = FrameBuffer(maxsize=args.buffer_size)
    
    # 创建并启动线程
    threads = []
    
    # 捕获线程
    capture_t = threading.Thread(target=capture_thread, args=(args, frame_buffer))
    capture_t.daemon = True
    threads.append(capture_t)
    
    # 2D姿态提取线程
    pose2d_t = threading.Thread(target=pose2d_thread, args=(args, frame_buffer))
    pose2d_t.daemon = True
    threads.append(pose2d_t)
    
    # 3D姿态预测线程
    pose3d_t = threading.Thread(target=pose3d_thread, args=(args, frame_buffer))
    pose3d_t.daemon = True
    threads.append(pose3d_t)
    
    # 3D姿态渲染线程
    render_t = threading.Thread(target=render_thread, args=(args, frame_buffer))
    render_t.daemon = True
    threads.append(render_t)
    
    # 显示线程
    display_t = threading.Thread(target=display_thread, args=(args, frame_buffer))
    display_t.daemon = True
    threads.append(display_t)
    
    # 启动所有线程
    print("启动在线3D姿态估计管道...")
    for t in threads:
        t.start()
        
    try:
        # 等待显示线程结束
        display_t.join()
    except KeyboardInterrupt:
        print("\n检测到键盘中断，正在退出...")
        stop_signal = True
        
    # 等待所有线程结束
    for t in threads:
        if t != display_t:
            t.join(timeout=1.0)
            
    print("程序已退出")


if __name__ == "__main__":
    main() 