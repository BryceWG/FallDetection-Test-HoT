import os
import sys
import cv2
import time
import copy
import torch
import argparse
import numpy as np
import threading
import queue
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 导入自定义模块
from utils import show2Dpose, show3Dpose
from data_processor import normalize_screen_coordinates, convert_yolov11_to_hrnet_keypoints, convert_yolov11_to_hrnet_scores, camera_to_world
from model.mixste.hot_mixste import Model

# 全局变量，用于线程间通信
frame_queue = queue.Queue(maxsize=5)  # 减小帧队列大小
keypoints_queue = queue.Queue(maxsize=5)  # 减小关键点队列大小
pose_3d_queue = queue.Queue(maxsize=3)  # 减小3D姿态队列大小
visualization_2d_queue = queue.Queue(maxsize=2)  # 减小2D可视化队列大小
visualization_3d_queue = queue.Queue(maxsize=1)  # 将3D可视化队列大小设为1，确保始终显示最新帧
stop_event = threading.Event()

class PoseEstimator:
    def __init__(self, args):
        """
        初始化姿态估计器
        
        参数:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualization_mode = args.visualization
        self.output_dir = args.output_dir
        self.save_results = True  # 默认启用结果保存
        self.frame_counter = 0
        self.fps_limit = args.video_fps_limit
        self.use_opencv_render = args.use_opencv_render  # 使用OpenCV渲染标志
        
        # 固定窗口批量处理相关变量
        self.use_fixed_window = True  # 默认使用固定窗口处理
        self.window_size = 243  # 固定窗口大小，由模型架构决定
        self.warmup_completed = False  # 预热完成标志
        self.warmup_frames_collected = 0  # 已收集的预热帧数
        self.current_window_frames = []  # 当前窗口收集的帧
        self.current_window_keypoints = []  # 当前窗口收集的关键点
        self.current_window_scores = []  # 当前窗口收集的分数
        self.window_processing = False  # 窗口处理中标志
        self.window_count = 0  # 已处理窗口计数
        self.last_window_completed_time = 0  # 上一个窗口完成时间
        self.window_processing_delay = 0  # 窗口处理延迟
        self.estimated_delay = 9.1  # 估计延迟（秒）
        self.is_shutting_down = False  # 关闭中标志
        
        # 创建3D预测结果存储
        self.all_window_predictions = []  # 存储所有窗口的3D预测结果
        self.current_display_index = 0  # 当前显示的帧索引
        self.total_frames_processed = 0  # 已处理的总帧数
        
        # 打印重要的配置信息
        print(f"\n=== System Configuration ===")
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Visualization Mode: {'Enabled' if self.visualization_mode else 'Disabled'}")
        print(f"3D Rendering Method: {'OpenCV (Fast)' if self.use_opencv_render else 'Matplotlib (Detailed)'}")
        print(f"Window Processing Mode: {'Fixed Window Batch Processing'}")
        print(f"Window Size: {self.window_size} frames")
        print(f"Estimated Delay: ~{self.estimated_delay} seconds")
        print(f"Save Results: Enabled")
        print(f"Output Directory: {self.output_dir}")
        if self.fps_limit > 0:
            print(f"Video FPS Limit: {self.fps_limit} FPS")
        print(f"==============\n")
        
        # 创建输出目录
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'input_2D'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'output_3D'), exist_ok=True)
            if self.visualization_mode:
                os.makedirs(os.path.join(self.output_dir, 'pose2D'), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, 'pose3D'), exist_ok=True)
        
        # 初始化模型
        self._init_yolo_model()
        self._init_mixste_model()
        
        # 摄像头设置
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0
        
        # 用于存储结果的数组
        self.all_keypoints = []
        self.all_scores = []
        self.all_poses_3d = []
        
        # 性能统计
        self.yolo_time = 0
        self.mixste_time = 0
        self.frame_time = 0
        self.last_3d_update = 0
        self.last_frame_time = time.time()
        
        # 设置3D可视化参数
        # OpenCV 3D渲染参数 - 无论是否可视化都需要初始化，因为保存结果时可能会用到
        self.render_size = (800, 800)  # 渲染尺寸
        self.render_background = (255, 255, 255)  # 白色背景
        self.render_scale = 150  # 缩放因子
        self.render_center = (400, 400)  # 中心点
        self.render_azimuth = 70  # 方位角(度)
        self.render_elevation = 15  # 仰角(度)
        
        if self.visualization_mode:
            if not self.use_opencv_render:
                # 使用原来的Matplotlib方式
                self.fig = plt.figure(figsize=(5, 5))
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.canvas = FigureCanvasAgg(self.fig)
            
        # 四元数旋转参数（用于3D坐标转换）
        self.rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], 
                            dtype='float32')
            
    def _init_yolo_model(self):
        """初始化YOLO模型"""
        print("Loading YOLO model...")
        try:
            self.yolo_model = YOLO(self.args.yolo_model)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
            
    def _init_mixste_model(self):
        """初始化MixSTE模型"""
        print("Loading MixSTE model...")
        try:
            # 配置模型参数
            self.mixste_args = self._get_mixste_args()
            
            # 加载模型
            self.mixste_model = Model(self.mixste_args).to(self.device)
            
            # 加载预训练权重
            model_path = self.args.mixste_model
            pre_dict = torch.load(model_path, map_location=self.device)
            model_dict = self.mixste_model.state_dict()
            state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.mixste_model.load_state_dict(model_dict)
            
            # 设置为评估模式
            self.mixste_model.eval()
            print("MixSTE model loaded successfully")
        except Exception as e:
            print(f"Error loading MixSTE model: {e}")
            sys.exit(1)
            
    def _get_mixste_args(self):
        """获取MixSTE模型参数"""
        args = argparse.Namespace()
        
        # 使用默认帧数
        frames = 243  # 默认使用243帧，与预训练模型匹配
            
        args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, frames
        args.token_num, args.layer_index = 81, 3
        args.pad = (args.frames - 1) // 2
        args.previous_dir = 'checkpoint/pretrained/hot_mixste'
        args.n_joints, args.out_joints = 17, 17
        return args
            
    def setup_camera(self):
        """设置摄像头"""
        print(f"Initializing camera (ID: {self.args.camera_id})...")
        self.cap = cv2.VideoCapture(self.args.camera_id)
        
        # 检查摄像头是否正常打开
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit(1)
            
        # 获取摄像头信息，不再手动设置分辨率
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized successfully, resolution: {self.frame_width}x{self.frame_height}")
        
    def release_resources(self):
        """释放资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released")
        
    def extract_2d_pose(self, frame):
        """
        从帧中提取2D姿态
        
        参数:
            frame: 输入图像帧
            
        返回:
            keypoints: 关键点坐标，形状为[17, 2]
            scores: 关键点置信度，形状为[17]
        """
        start_time = time.time()
        
        # 使用YOLO进行检测
        results = self.yolo_model(frame, verbose=False)
        
        # 检查是否检测到人体
        if len(results) > 0 and results[0].keypoints is not None:
            # 获取第一个检测到的人的关键点
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # [17, 3] 数组，每行是[x, y, conf]
            
            # 分离坐标和置信度
            kpts_xy = keypoints[:, :2]  # [17, 2]
            scores = keypoints[:, 2]  # [17]
            
            # 应用置信度阈值过滤
            if len(self.all_keypoints) > 0:
                try:
                    # 对于低置信度的关键点，使用上一帧的对应关键点
                    low_conf_mask = scores < self.args.conf_thresh
                    if np.any(low_conf_mask) and len(self.all_keypoints[-1]) == len(low_conf_mask):
                        kpts_xy[low_conf_mask] = self.all_keypoints[-1][low_conf_mask]
                        scores[low_conf_mask] = self.all_scores[-1][low_conf_mask]
                except Exception as e:
                    print(f"Error applying confidence threshold: {e}")
                    # 如果出错，不应用任何过滤
        else:
            # 如果没有检测到人，使用上一帧的关键点或零填充
            if len(self.all_keypoints) > 0:
                kpts_xy = self.all_keypoints[-1].copy()  # 使用copy以避免修改原始数据
                scores = self.all_scores[-1].copy()
            else:
                kpts_xy = np.zeros((17, 2))
                scores = np.zeros(17)
                
        self.yolo_time = time.time() - start_time
        return kpts_xy, scores
        
    def predict_3d_pose(self, input_keypoints):
        """
        预测3D姿态
        
        参数:
            input_keypoints: 输入关键点，形状为[F, 17, 2]，F是帧数
            
        返回:
            output_3d: 3D姿态预测结果，形状为[17, 3]
        """
        start_time = time.time()
        
        window_size = self.mixste_args.frames
        
        # 确保输入的帧数等于窗口大小
        assert len(input_keypoints) == window_size, f"输入帧数应为{window_size}，实际为{len(input_keypoints)}"
        
        # 将输入关键点转换为需要的格式
        input_keypoints_array = np.array(input_keypoints)[np.newaxis, ...]  # [1, F, 17, 2]
        # 将YOLO关键点转换为HRNet格式
        input_keypoints_converted = convert_yolov11_to_hrnet_keypoints(input_keypoints_array.copy())
        # 使用转换后的关键点，取第一个批次
        input_keypoints = input_keypoints_converted[0]
            
        # 归一化坐标
        input_2D = normalize_screen_coordinates(input_keypoints, w=self.frame_width, h=self.frame_height)
        
        # 数据增强（左右翻转）
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        
        # 准备输入张量
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        input_2D = input_2D[np.newaxis, :, :, :, :]
        input_2D = torch.from_numpy(input_2D.astype('float32')).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            output_3D_non_flip = self.mixste_model(input_2D[:, 0])
            output_3D_flip = self.mixste_model(input_2D[:, 1])
            
            # 处理翻转的预测结果
            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
            
            # 融合正常和翻转的预测结果
            output_3D = (output_3D_non_flip + output_3D_flip) / 2
            
        # 取中间帧的预测结果（最准确的部分）
        mid_frame = window_size // 2
        output_3d = output_3D[0, mid_frame].cpu().detach().numpy()
        output_3d[0, :] = 0  # 将根节点置零
        
        self.mixste_time = time.time() - start_time
        return output_3d
        
    def create_2d_visualization(self, frame, keypoints_2d):
        """
        创建2D可视化
        
        参数:
            frame: 原始图像帧
            keypoints_2d: 2D关键点，形状为[17, 2]
            
        返回:
            vis_2d: 2D可视化图像
        """
        # 创建2D姿态可视化
        vis_2d = frame.copy()
        vis_2d = show2Dpose(keypoints_2d, vis_2d)
        
        # 添加性能信息
        fps_text = f"System Frame Rate: {1.0/self.frame_time:.1f} FPS"
        yolo_text = f"YOLO Processing: {1.0/self.yolo_time:.1f} FPS"
        cv2.putText(vis_2d, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_2d, yolo_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_2d
        
    def create_3d_visualization(self, pose_3d):
        """
        创建3D可视化
        
        参数:
            pose_3d: 3D姿态，形状为[17, 3]
            
        返回:
            vis_3d: 3D可视化图像
        """
        if self.use_opencv_render:
            return self.create_3d_visualization_opencv(pose_3d)
        else:
            # 使用原来的Matplotlib方式
            # 创建3D姿态可视化
            self.ax.clear()
            pose_3d_world = camera_to_world(pose_3d.copy(), R=self.rot, t=0)
            pose_3d_world[:, 2] -= np.min(pose_3d_world[:, 2])
            show3Dpose(pose_3d_world, self.ax, fix_z=self.args.fix_z)
            
            # 设置3D视图
            self.ax.view_init(elev=15., azim=70)
            
            # 渲染3D图形到numpy数组
            self.canvas.draw()
            vis_3d = np.array(self.canvas.buffer_rgba())
            vis_3d = cv2.cvtColor(vis_3d, cv2.COLOR_RGBA2BGR)
            
            # 添加性能信息
            mixste_text = f"3D Prediction: {1.0/self.mixste_time:.1f} FPS"
            # 在图像底部添加性能信息
            cv2.putText(vis_3d, mixste_text, (10, vis_3d.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return vis_3d
            
    def create_3d_visualization_opencv(self, pose_3d):
        """
        使用OpenCV直接绘制简化的3D模型
        
        参数:
            pose_3d: 3D姿态，形状为[17, 3]
            
        返回:
            img: 3D可视化图像
        """
        try:
            # 预先测量渲染时间开始
            render_start = time.time()
            
            # 确保必要的属性存在，如果不存在则使用默认值
            render_size = getattr(self, 'render_size', (800, 800))
            render_scale = getattr(self, 'render_scale', 150)
            render_center = getattr(self, 'render_center', (400, 400))
            render_azimuth = getattr(self, 'render_azimuth', 70)
            render_elevation = getattr(self, 'render_elevation', 15)
            
            # 坐标转换
            pose_3d_world = camera_to_world(pose_3d.copy(), R=self.rot, t=0)
            pose_3d_world[:, 2] -= np.min(pose_3d_world[:, 2])
            
            # 创建空白图像
            img = np.ones((render_size[0], render_size[1], 3), dtype=np.uint8) * 255
            
            # 添加网格（更好的3D感知）- 在绘制骨架前绘制网格
            grid_step = 50
            grid_size = 1  # 减小网格线宽度
            grid_color = (240, 240, 240)  # 更浅的网格颜色
            
            # 绘制水平线和垂直线形成网格
            for i in range(0, img.shape[0], grid_step):
                cv2.line(img, (0, i), (img.shape[1], i), grid_color, grid_size)
                
            for i in range(0, img.shape[1], grid_step):
                cv2.line(img, (i, 0), (i, img.shape[0]), grid_color, grid_size)
            
            # 3D到2D的投影参数
            scale = render_scale
            center_x, center_y = render_center
            
            # 应用旋转矩阵 (简化版)
            theta = np.radians(render_azimuth)  # azimuth
            phi = np.radians(render_elevation)    # elevation
            
            # 一次性计算所有关键点2D坐标，提高效率
            points_2d = []
            for point in pose_3d_world:
                # 简化的3D到2D投影
                x = point[0] * np.cos(theta) - point[1] * np.sin(theta)
                y = point[2] * np.cos(phi) - (point[0] * np.sin(theta) + point[1] * np.cos(theta)) * np.sin(phi)
                
                # 缩放和平移
                x = int(x * scale + center_x)
                y = int(y * scale + center_y)
                points_2d.append((x, y))
            
            # 绘制坐标系参考轴 (X: 红色, Y: 绿色, Z: 蓝色) - 先绘制参考轴
            origin = points_2d[0]  # 使用根关节点作为原点
            axis_length = 50
            
            # X轴
            x_axis = (int(origin[0] + axis_length * np.cos(theta)), 
                     int(origin[1] - axis_length * np.sin(phi) * np.sin(theta)))
            cv2.line(img, origin, x_axis, (0, 0, 255), 2)  # 红色
            cv2.putText(img, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Y轴
            y_axis = (int(origin[0] - axis_length * np.sin(theta)), 
                     int(origin[1] - axis_length * np.sin(phi) * np.cos(theta)))
            cv2.line(img, origin, y_axis, (0, 255, 0), 2)  # 绿色
            cv2.putText(img, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Z轴
            z_axis = (int(origin[0]), int(origin[1] + axis_length * np.cos(phi)))
            cv2.line(img, origin, z_axis, (255, 0, 0), 2)  # 蓝色
            cv2.putText(img, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 绘制骨架连接
            connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                          [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                          [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
                          
            # 定义颜色: 绿色(左侧), 黄色(右侧), 蓝色(中线)
            colors = [(138, 201, 38),    # 绿色 - 左侧
                      (25, 130, 196),    # 蓝色 - 中线
                      (255, 202, 58)]    # 黄色 - 右侧
                      
            # 定义左右侧: 1=左侧, 2=右侧, 3=中线
            LR = [2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2]
            
            # 绘制连接线
            for j, c in enumerate(connections):
                pt1 = points_2d[c[0]]
                pt2 = points_2d[c[1]]
                # 确保点在图像范围内
                if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and 
                    0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                    cv2.line(img, pt1, pt2, colors[LR[j]-1], 3)
            
            # 绘制关节点 - 在连线后绘制，使其位于顶层
            for i, point in enumerate(points_2d):
                if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
                    cv2.circle(img, point, 5, (0, 0, 255), -1)
            
            # 计算实际渲染时间
            current_render_time = time.time() - render_start
            
            # 更新时间记录
            last_3d_update = getattr(self, 'last_3d_update', time.time())
            
            # 计算实际更新间隔
            time_since_last_update = time.time() - last_3d_update
            
            # 添加性能信息到图像
            mixste_time = getattr(self, 'mixste_time', 0.1)
            mixste_text = f"3D Prediction: {1.0/max(0.001, mixste_time):.1f} FPS"
            render_text = f"Render Method: OpenCV ({current_render_time*1000:.1f}ms)"
            update_text = f"Last Update: {time_since_last_update:.2f}s ago"
            refresh_rate = f"Refresh Rate: {1.0/max(0.01, time_since_last_update):.1f} FPS"
            render_fps_text = f"Pure Render: {1.0/max(0.001, current_render_time):.1f} FPS"
            
            # 添加半透明背景，使文字更容易阅读
            overlay = img.copy()
            cv2.rectangle(overlay, (5, img.shape[0]-145), (350, img.shape[0]-5), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # 添加文字
            cv2.putText(img, mixste_text, (10, img.shape[0] - 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, render_text, (10, img.shape[0] - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, update_text, (10, img.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, refresh_rate, (10, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, render_fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 添加更新时间戳
            timestamp = f"Update Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
            cv2.putText(img, timestamp, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return img
            
        except Exception as e:
            print(f"OpenCV 3D rendering error: {e}")
            # 创建一个显示错误信息的空白图像
            img = np.ones((800, 800, 3), dtype=np.uint8) * 255
            error_text = f"3D Rendering Error: {str(e)}"
            cv2.putText(img, error_text, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img
        
    def save_results_to_file(self):
        """保存结果到文件"""
        if not self.save_results:
            return
            
        print("Saving results to file...")
        
        # 固定窗口批处理模式
        if len(self.all_window_predictions) == 0:
            print("No predictions available")
            return
            
        # 收集所有窗口的关键点和3D姿态
        all_keypoints = []
        all_scores = []
        all_poses_3d = []
        
        for window in self.all_window_predictions:
            all_keypoints.extend(window['keypoints'])
            all_scores.extend(window['scores'])
            
            # 对于3D姿态，目前每个窗口只有一个输出
            # 复制这个姿态窗口大小次，以匹配帧数
            window_poses = [window['pose_3d']] * len(window['keypoints'])
            all_poses_3d.extend(window_poses)
            
        # 转换关键点到HRNet格式
        keypoints_array = np.array(all_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
        scores_array = np.array(all_scores)[np.newaxis, ...]  # [1, T, 17]
        
        # 转换格式
        keypoints_hrnet = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
        scores_hrnet = convert_yolov11_to_hrnet_scores(scores_array.copy())
        
        # 保存2D关键点
        output_2d_path = os.path.join(self.output_dir, 'input_2D', 'input_keypoints_2d.npz')
        np.savez_compressed(output_2d_path, reconstruction=keypoints_hrnet)
        
        # 保存3D姿态
        if len(all_poses_3d) > 0:
            output_3d_path = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
            poses_3d_array = np.array(all_poses_3d)
            np.savez_compressed(output_3d_path, reconstruction=poses_3d_array)
            
        print(f"Saved {len(all_keypoints)} frames of results to {self.output_dir}")

    def extract_2d_pose_thread(self):
        """2D姿态提取线程"""
        last_update = time.time()
        
        while not stop_event.is_set():
            try:
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = frame_queue.get()
                
                try:
                    keypoints, scores = self.extract_2d_pose(frame)
                except Exception as e:
                    print(f"Error extracting 2D pose: {e}")
                    # 出错时创建空关键点
                    keypoints = np.zeros((17, 2))
                    scores = np.zeros(17)
                
                # 将关键点格式转换为HRNet格式，方便后续处理和可视化
                keypoints_array = np.array([keypoints])[np.newaxis, ...]  # [1, 1, 17, 2]
                
                # 检查关键点是否有效（所有值都为0表示无效）
                is_valid_keypoints = not np.all(keypoints == 0)
                
                if is_valid_keypoints:
                    # 只在关键点有效时进行转换
                    keypoints_converted = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
                    keypoints_vis = keypoints_converted[0, 0]  # 获取转换后的单帧关键点 [17, 2]
                else:
                    # 对于无效关键点，创建一个空的可视化关键点集
                    keypoints_vis = np.zeros((17, 2))
                
                # 将结果放入队列
                if not keypoints_queue.full():
                    keypoints_queue.put((frame, keypoints, scores))
                    
                # 创建2D可视化并放入队列
                if self.visualization_mode and not visualization_2d_queue.full():
                    vis_2d = frame.copy()
                    if is_valid_keypoints:
                        try:
                            vis_2d = self.create_2d_visualization(frame, keypoints_vis)
                        except Exception as e:
                            print(f"Error creating 2D visualization: {e}")
                            # 显示错误信息在原始帧上
                            error_text = f"Visualization Error: {str(e)}"
                            cv2.putText(vis_2d, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # 没有检测到人，只显示原始帧和状态信息
                        fps_text = f"System Frame Rate: {1.0/max(0.001, self.frame_time):.1f} FPS"
                        yolo_text = f"YOLO Processing: {1.0/max(0.001, self.yolo_time):.1f} FPS"
                        status_text = "Status: No Person Detected"
                        cv2.putText(vis_2d, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, yolo_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    visualization_2d_queue.put(vis_2d)
                    
                # 记录关键点用于保存
                if self.save_results:
                    self.all_keypoints.append(keypoints)
                    self.all_scores.append(scores)
                    
            except Exception as e:
                print(f"Error in 2D pose extraction thread: {e}")
                import traceback
                traceback.print_exc()
                # 添加短暂休眠，避免错误发生时CPU使用率过高
                time.sleep(0.1)
                
    def predict_3d_pose_thread(self):
        """3D姿态预测线程"""
        buffer_keypoints = []
        update_interval = 0.02  # 固定的更新间隔，约50FPS
        last_update = time.time()
        render_time = 0.0  # 用于记录渲染时间
        
        # 性能监控变量
        fps_history = []
        
        while not stop_event.is_set():
            try:
                if keypoints_queue.empty():
                    # 更积极地检查队列
                    time.sleep(0.005)  # 减少等待时间
                    continue
                    
                frame, keypoints, scores = keypoints_queue.get()
                
                # 检查关键点是否有效（所有值都为0表示无效）
                is_valid_keypoints = not np.all(keypoints == 0)
                
                if is_valid_keypoints:
                    # 添加到缓冲区
                    buffer_keypoints.append(keypoints)
                    
                    # 管理缓冲区大小，保留最新的帧
                    if len(buffer_keypoints) > self.mixste_args.frames:
                        # 如果缓冲区超过需要的帧数，删除最旧的帧
                        buffer_keypoints = buffer_keypoints[-self.mixste_args.frames:]
                        
                    # 控制3D预测的更新频率
                    current_time = time.time()
                    time_since_last_update = current_time - last_update
                    
                    # 确保有足够的帧用于初始预测
                    min_frames_needed = min(10, self.mixste_args.frames // 16)  # 减少所需最小帧数
                    
                    # 条件1：超过更新间隔；条件2：有足够的帧；条件3：避免队列满
                    if ((time_since_last_update >= update_interval and 
                        len(buffer_keypoints) >= min_frames_needed) or 
                        time_since_last_update >= 0.5):  # 强制不超过0.5秒更新一次
                        
                        render_start = time.time()
                        
                        # 准备输入数据
                        input_keypoints = np.array(buffer_keypoints)
                        
                        # 如果帧数不足，使用复制策略填充
                        if len(input_keypoints) < self.mixste_args.frames:
                            # 计算需要填充的帧数
                            pad_length = self.mixste_args.frames - len(input_keypoints)
                            
                            # 使用现有帧的镜像填充策略，使得时间连续性更自然
                            if len(input_keypoints) > 1:
                                # 前向填充：复制第一帧并在其前面添加微小随机噪声以增加稳定性
                                pad_front = np.repeat(input_keypoints[:1], pad_length // 2, axis=0)
                                # 为填充帧添加微小的随机扰动，使得预测更稳定
                                pad_front += np.random.normal(0, 0.2, pad_front.shape)  # 减少噪声幅度
                                
                                # 后向填充：复制最后一帧
                                pad_back = np.repeat(input_keypoints[-1:], pad_length - pad_length // 2, axis=0)
                                
                                # 组合所有帧
                                input_keypoints = np.concatenate([pad_front, input_keypoints, pad_back], axis=0)
                            else:
                                # 如果只有一帧，则前后都使用该帧进行填充
                                pad_keypoints = np.repeat(input_keypoints, pad_length, axis=0)
                                input_keypoints = np.concatenate([input_keypoints, pad_keypoints], axis=0)
                        
                        # 预测3D姿态
                        pose_3d = self.predict_3d_pose(input_keypoints)
                        
                        # 将结果放入队列
                        if not pose_3d_queue.full():
                            # 使用最新的一帧关键点(已转换格式)和3D姿态
                            keypoints_array = np.array([buffer_keypoints[-1]])[np.newaxis, ...]  # [1, 1, 17, 2]
                            keypoints_converted = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
                            keypoints_vis = keypoints_converted[0, 0]  # 最新一帧的转换后关键点
                            pose_3d_queue.put((frame, keypoints_vis, pose_3d))
                            
                        # 创建3D可视化并放入队列
                        if self.visualization_mode:
                            # 清空可视化队列，确保始终显示最新的渲染结果
                            with visualization_3d_queue.mutex:
                                visualization_3d_queue.queue.clear()
                                
                            vis_3d = self.create_3d_visualization(pose_3d)
                            visualization_3d_queue.put(vis_3d)
                            
                            # 计算并记录渲染时间
                            render_time = time.time() - render_start
                            
                            # 计算实际更新帧率
                            if time_since_last_update > 0:
                                current_fps = 1.0 / time_since_last_update
                                fps_history.append(current_fps)
                                # 保留最近10个值计算平均帧率
                                if len(fps_history) > 10:
                                    fps_history.pop(0)
                        
                        # 记录3D姿态用于保存
                        if self.save_results:
                            self.all_poses_3d.append(pose_3d)
                            
                        last_update = current_time
                        self.last_3d_update = current_time
                else:
                    # 如果没有有效关键点，清空可视化队列并放入一个空的3D可视化
                    if self.visualization_mode:
                        # 清空队列
                        with visualization_3d_queue.mutex:
                            visualization_3d_queue.queue.clear()
                            
                        # 创建一个显示"未检测到人体"的空白3D可视化
                        blank_3d = np.ones((800, 800, 3), dtype=np.uint8) * 255
                        status_text = "Status: No Person Detected"
                        cv2.putText(blank_3d, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        fps_text = f"3D Rendering: {1.0/max(0.001, render_time):.1f} FPS"
                        avg_fps_text = f"Average Update Rate: {np.mean(fps_history) if fps_history else 0:.1f} FPS"
                        cv2.putText(blank_3d, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(blank_3d, avg_fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        visualization_3d_queue.put(blank_3d)
                
            except Exception as e:
                print(f"Error in 3D pose prediction thread: {e}")
                import traceback
                traceback.print_exc()
                
    def process_fixed_window_thread(self):
        """固定窗口批量处理线程"""
        while not stop_event.is_set():
            try:
                # 收集关键点直到达到窗口大小
                while len(self.current_window_keypoints) < self.window_size and not stop_event.is_set():
                    if keypoints_queue.empty():
                        time.sleep(0.01)
                        continue
                        
                    frame, keypoints, scores = keypoints_queue.get()
                    
                    # 检查关键点是否有效（所有值都非零）
                    is_valid_keypoints = not np.all(keypoints == 0)
                    
                    # 只添加有效的关键点到窗口
                    if is_valid_keypoints:
                        self.current_window_frames.append(frame)
                        self.current_window_keypoints.append(keypoints)
                        self.current_window_scores.append(scores)
                        self.warmup_frames_collected += 1
                        
                        # 可视化当前收集阶段的2D姿态
                        if self.visualization_mode and not visualization_2d_queue.full():
                            # 创建收集状态可视化
                            vis_frame = frame.copy()
                            keypoints_array = np.array([keypoints])[np.newaxis, ...]
                            keypoints_vis = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())[0, 0]
                            
                            # 绘制骨架
                            try:
                                vis_frame = self.create_2d_visualization(frame, keypoints_vis)
                            except Exception as e:
                                print(f"Error creating 2D visualization: {e}")
                                error_text = f"Visualization Error: {str(e)}"
                                cv2.putText(vis_frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # 添加收集状态信息
                            status_text = f"Collecting Frames: {len(self.current_window_keypoints)}/{self.window_size}"
                            cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # 显示预热状态
                            if not self.warmup_completed:
                                warmup_text = f"System Warming Up: {self.warmup_frames_collected}/{self.window_size}"
                                cv2.putText(vis_frame, warmup_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                            visualization_2d_queue.put(vis_frame)
                    else:
                        # 检测到无效关键点，在可视化中显示状态
                        if self.visualization_mode and not visualization_2d_queue.full():
                            vis_frame = frame.copy()
                            status_text = "No Person Detected, Waiting for Valid Input..."
                            cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # 显示当前收集状态
                            collection_text = f"Frames Collected: {len(self.current_window_keypoints)}/{self.window_size}"
                            cv2.putText(vis_frame, collection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                            
                            visualization_2d_queue.put(vis_frame)
                            
                    # 如果正在关闭，并且已填充足够的帧，则退出收集循环
                    if self.is_shutting_down and len(self.current_window_keypoints) == self.window_size:
                        print("During shutdown, last window filled")
                        break
                
                # 设置状态为正在处理窗口
                self.window_processing = True
                
                # 只有当收集了足够的帧时才处理
                if len(self.current_window_keypoints) == self.window_size:
                    try:
                        # 记录窗口开始处理时间
                        window_start_time = time.time()
                        
                        # 使用收集的关键点进行3D姿态预测
                        pose_3d = self.predict_3d_pose(self.current_window_keypoints)
                        
                        # 计算窗口处理延迟
                        self.window_processing_delay = time.time() - window_start_time
                        
                        # 创建窗口预测结果
                        window_result = {
                            'window_id': self.window_count,
                            'frames': self.current_window_frames,
                            'keypoints': self.current_window_keypoints,
                            'scores': self.current_window_scores,
                            'pose_3d': pose_3d,
                            'processing_delay': self.window_processing_delay
                        }
                        
                        # 添加到所有窗口预测结果
                        self.all_window_predictions.append(window_result)
                        
                        # 更新总帧数
                        self.total_frames_processed += len(self.current_window_keypoints)
                        
                        # 创建3D可视化
                        if self.visualization_mode and not visualization_3d_queue.full():
                            # 为每个帧创建可视化，但都使用相同的3D姿态
                            for i in range(len(self.current_window_frames)):
                                try:
                                    vis_3d = self.create_3d_visualization_fixed_window(
                                        pose_3d, 
                                        self.window_count,
                                        i,
                                        self.total_frames_processed - self.window_size + i
                                    )
                                    visualization_3d_queue.put(vis_3d)
                                except Exception as e:
                                    print(f"Error creating 3D visualization: {e}")
                                    # 创建一个错误信息图像
                                    error_img = np.ones((800, 800, 3), dtype=np.uint8) * 255
                                    error_text = f"3D Visualization Error: {str(e)}"
                                    cv2.putText(error_img, error_text, (10, 400), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    visualization_3d_queue.put(error_img)
                        
                        # 增加窗口计数
                        self.window_count += 1
                        
                        # 设置预热完成标志
                        if not self.warmup_completed:
                            self.warmup_completed = True
                            print(f"\nWarm-up complete! Processed first complete window ({self.window_size} frames).")
                            print(f"Window processing delay: {self.window_processing_delay:.2f} seconds")
                        
                        # 更新最后一个窗口完成时间
                        self.last_window_completed_time = time.time()
                        
                    except Exception as e:
                        print(f"Error processing window: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # 清空当前窗口
                        self.current_window_frames = []
                        self.current_window_keypoints = []
                        self.current_window_scores = []
                        self.window_processing = False
                        
                        # 如果正在关闭，设置关闭标志为False，表示处理完成
                        if self.is_shutting_down:
                            print("Last window processing complete, ready to close...")
                            self.is_shutting_down = False
                
            except Exception as e:
                print(f"Error in fixed window processing thread: {e}")
                import traceback
                traceback.print_exc()
                # 清空当前窗口，避免卡在错误状态
                self.current_window_frames = []
                self.current_window_keypoints = []
                self.current_window_scores = []
                self.window_processing = False
                time.sleep(0.1)  # 添加短暂休眠

    def create_3d_visualization_fixed_window(self, pose_3d, window_id, frame_in_window, total_frame_index):
        """
        为固定窗口模式创建3D可视化
        
        参数:
            pose_3d: 3D姿态，形状为[17, 3]
            window_id: 窗口ID
            frame_in_window: 窗口内的帧索引
            total_frame_index: 总帧索引
            
        返回:
            img: 3D可视化图像
        """
        try:
            # 调用现有的OpenCV渲染函数
            img = self.create_3d_visualization_opencv(pose_3d)
            
            # 添加窗口处理信息
            delay_text = f"System Delay: ~{self.estimated_delay:.1f} seconds"
            window_text = f"Window #{window_id+1}, Frame {frame_in_window+1}/{self.window_size}"
            total_text = f"Total Frames: {total_frame_index+1}"
            
            # 添加半透明背景区域，用于窗口信息
            overlay = img.copy()
            cv2.rectangle(overlay, (5, 5), (400, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # 添加信息文本
            cv2.putText(img, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, window_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, total_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
            
            return img
        except Exception as e:
            print(f"Error creating fixed window 3D visualization: {e}")
            # 创建一个显示错误信息的空白图像
            img = np.ones((800, 800, 3), dtype=np.uint8) * 255
            error_text = f"3D Visualization Error: {str(e)}"
            cv2.putText(img, error_text, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

    def run(self):
        """运行姿态估计pipeline"""
        self.setup_camera()
        
        # 启动工作线程
        threads = []
        threads.append(threading.Thread(target=self.extract_2d_pose_thread))
        
        # 使用固定窗口批量处理模式
        threads.append(threading.Thread(target=self.process_fixed_window_thread))
        print("Using fixed window batch processing mode")
            
        for thread in threads:
            thread.daemon = True
            thread.start()
            
        try:
            print("Starting real-time pose estimation...")
            print("Press 'q' to exit")
            
            print("\nFixed Window Batch Processing Configuration:")
            print(f"- Window Size: {self.window_size} frames")
            print(f"- Estimated Delay: ~{self.estimated_delay} seconds")
            print(f"- Processing Method: Batch processing every {self.window_size} frames")
            print("- First use requires warm-up process, please wait for the first window to complete\n")

            # 主循环
            while True:
                start_time = time.time()
                
                # 帧率控制
                if self.fps_limit > 0:
                    time_since_last_frame = time.time() - self.last_frame_time
                    target_time_between_frames = 1.0 / self.fps_limit
                    
                    if time_since_last_frame < target_time_between_frames:
                        # 等待以确保不超过目标帧率
                        time.sleep(target_time_between_frames - time_since_last_frame)
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("Cannot read frame from camera")
                    break
                
                self.last_frame_time = time.time()
                self.frame_counter += 1
                    
                # 添加到帧队列
                if not frame_queue.full():
                    frame_queue.put(frame)
                    
                # 显示原始摄像头画面（不包含姿态标识）
                cv2.imshow("Camera Feed", frame)
                
                # 显示2D可视化结果
                if self.visualization_mode and not visualization_2d_queue.empty():
                    vis_2d = visualization_2d_queue.get()
                    cv2.imshow("2D Pose Detection", vis_2d)
                    
                # 显示3D可视化结果
                if self.visualization_mode and not visualization_3d_queue.empty():
                    vis_3d = visualization_3d_queue.get()
                    cv2.imshow("3D Pose Reconstruction", vis_3d)
                    
                # 计算帧率
                self.frame_time = time.time() - start_time
                
                # 显示性能信息
                fps = 1.0 / max(0.001, self.frame_time)  # 避免除以零
                if self.frame_counter % 30 == 0:  # 每30帧更新一次控制台输出
                    print(f"\rDisplay Rate: {fps:.1f} FPS | YOLO: {1.0/max(0.001, self.yolo_time):.1f} FPS | " +
                          f"Window: {self.window_count} | Frames: {self.total_frames_processed}", end="")
                
                # 设置更短的等待时间，提高响应速度
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested exit")
                    
                    # 处理剩余的帧
                    if len(self.current_window_keypoints) > 0:
                        print(f"\nWaiting to process remaining {len(self.current_window_keypoints)} frames...")
                        self.is_shutting_down = True
                        
                        # 如果有部分收集的帧但不足一个窗口，填充剩余帧
                        if 0 < len(self.current_window_keypoints) < self.window_size:
                            frames_needed = self.window_size - len(self.current_window_keypoints)
                            print(f"Filling {frames_needed} frames to complete the last window")
                            
                            # 使用最后一帧的复制来填充
                            last_frame = self.current_window_frames[-1]
                            last_keypoints = self.current_window_keypoints[-1]
                            last_scores = self.current_window_scores[-1]
                            
                            for _ in range(frames_needed):
                                self.current_window_frames.append(last_frame.copy())
                                self.current_window_keypoints.append(last_keypoints.copy())
                                self.current_window_scores.append(last_scores.copy())
                        
                        # 等待最后一个窗口处理完成
                        wait_start = time.time()
                        while self.is_shutting_down and time.time() - wait_start < 30:  # 最多等待30秒
                            if len(self.current_window_keypoints) == 0:
                                print("All windows processed, closing program...")
                                break
                            time.sleep(0.1)
                    
                    break
                    
        except KeyboardInterrupt:
            print("\nUser interrupted program")
        finally:
            # 停止所有线程
            stop_event.set()
            
            # 保存结果
            self.save_results_to_file()
            
            # 等待线程结束
            for thread in threads:
                thread.join(timeout=1.0)
                
            # 释放资源
            self.release_resources()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Real-time Pose Estimation')
    
    # 基本参数
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # 模型参数
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO model path')
    parser.add_argument('--mixste_model', type=str, default='checkpoint/pretrained/hot_mixste/model.pth', 
                       help='MixSTE model path')
    
    # 可视化参数
    parser.add_argument('--visualization', action='store_true', help='Enable visualization mode')
    parser.add_argument('--fix_z', action='store_true', help='Fix Z-axis range')
    parser.add_argument('--use_opencv_render', action='store_true', help='Use OpenCV for fast 3D rendering')
    
    # 其他参数
    parser.add_argument('--conf_thresh', type=float, default=0.5, 
                       help='Keypoint confidence threshold, keypoints below this value will use the corresponding keypoint from the previous frame')
    parser.add_argument('--video_fps_limit', type=int, default=30,
                       help='Limit processing frame rate to avoid system overload, set to 0 for no limit')
    
    return parser.parse_args()
    

def main():
    """主函数"""
    args = parse_args()
    estimator = PoseEstimator(args)
    estimator.run()

if __name__ == "__main__":
    main() 