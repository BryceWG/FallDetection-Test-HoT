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
        self.save_results = args.save_results
        self.frame_counter = 0
        self.fps_limit = args.video_fps_limit
        self.use_opencv_render = args.use_opencv_render  # 使用OpenCV渲染标志
        
        # 视频相关参数
        self.input_type = 'camera' if args.video_path is None else 'video'
        self.video_path = args.video_path
        self.video_loop = args.video_loop
        self.pause_video = False
        self.original_video_fps = 0  # 原始视频帧率，后续会读取
        
        # 固定窗口批量处理相关变量
        self.use_fixed_window = args.use_fixed_window
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
        print(f"\n=== 系统配置 ===")
        print(f"设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"输入类型: {'本地视频' if self.input_type == 'video' else '摄像头'}")
        if self.input_type == 'video':
            print(f"视频路径: {self.video_path}")
            print(f"循环播放: {'启用' if self.video_loop else '禁用'}")
        print(f"可视化模式: {'启用' if self.visualization_mode else '禁用'}")
        print(f"3D渲染方式: {'OpenCV (快速)' if self.use_opencv_render else 'Matplotlib (详细)'}")
        print(f"窗口处理模式: {'固定窗口批量处理' if self.use_fixed_window else '滑动窗口连续处理'}")
        if self.use_fixed_window:
            print(f"窗口大小: {self.window_size} 帧")
            print(f"估计延迟: 约 {self.estimated_delay} 秒")
        else:
            print(f"3D更新最小帧率: {args.min_3d_update_fps} FPS")
        print(f"结果保存: {'启用' if self.save_results else '禁用'}")
        if self.save_results:
            print(f"输出目录: {self.output_dir}")
        if self.fps_limit > 0:
            print(f"视频帧率限制: {self.fps_limit} FPS")
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
        if self.visualization_mode:
            if not self.use_opencv_render:
                # 使用原来的Matplotlib方式
                self.fig = plt.figure(figsize=(5, 5))
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.canvas = FigureCanvasAgg(self.fig)
            else:
                # OpenCV 3D渲染参数
                self.render_size = (800, 800)  # 渲染尺寸
                self.render_background = (255, 255, 255)  # 白色背景
                self.render_scale = 150  # 缩放因子
                self.render_center = (400, 400)  # 中心点
                self.render_azimuth = 70  # 方位角(度)
                self.render_elevation = 15  # 仰角(度)
            
        # 四元数旋转参数（用于3D坐标转换）
        self.rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], 
                            dtype='float32')
            
    def _init_yolo_model(self):
        """初始化YOLO模型"""
        print("正在加载YOLO模型...")
        try:
            self.yolo_model = YOLO(self.args.yolo_model)
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"加载YOLO模型时出错: {e}")
            sys.exit(1)
            
    def _init_mixste_model(self):
        """初始化MixSTE模型"""
        print("正在加载MixSTE模型...")
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
            print("MixSTE模型加载成功")
        except Exception as e:
            print(f"加载MixSTE模型时出错: {e}")
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
        """设置摄像头或视频输入"""
        if self.input_type == 'camera':
            print(f"正在初始化摄像头 (ID: {self.args.camera_id})...")
            self.cap = cv2.VideoCapture(self.args.camera_id)
            
            # 检查摄像头是否正常打开
            if not self.cap.isOpened():
                print("无法打开摄像头")
                sys.exit(1)
                
            # 获取摄像头信息
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = -1  # 摄像头没有固定帧数
            
            print(f"摄像头初始化成功，分辨率: {self.frame_width}x{self.frame_height}, FPS: {self.original_video_fps}")
            
        else:  # 视频文件输入
            print(f"正在加载视频文件: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            
            # 检查视频是否正常打开
            if not self.cap.isOpened():
                print(f"无法打开视频文件: {self.video_path}")
                sys.exit(1)
                
            # 获取视频信息
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = self.frame_count / self.original_video_fps
            
            # 设置视频开始时间
            if self.args.video_start_time > 0:
                start_frame = int(self.args.video_start_time * self.original_video_fps)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                print(f"视频将从 {self.args.video_start_time:.2f} 秒处开始播放")
                
            print(f"视频加载成功，总帧数: {self.frame_count}, 时长: {self.video_duration:.2f}秒")
            print(f"分辨率: {self.frame_width}x{self.frame_height}, FPS: {self.original_video_fps}")
            
        # 如果未指定帧率限制，使用原始视频帧率
        if self.fps_limit == 0:
            self.fps_limit = self.original_video_fps
            print(f"未指定帧率限制，将使用原始视频帧率: {self.fps_limit} FPS")
        
    def release_resources(self):
        """释放资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")
        
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
                # 对于低置信度的关键点，使用上一帧的对应关键点
                low_conf_mask = scores < self.args.conf_thresh
                if np.any(low_conf_mask):
                    kpts_xy[low_conf_mask] = self.all_keypoints[-1][low_conf_mask]
                    scores[low_conf_mask] = self.all_scores[-1][low_conf_mask]
        else:
            # 如果没有检测到人，使用上一帧的关键点或零填充
            if len(self.all_keypoints) > 0:
                kpts_xy = self.all_keypoints[-1]
                scores = self.all_scores[-1]
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
        
        # 检查是否使用固定窗口模式
        if self.use_fixed_window:
            # 确保输入的帧数等于窗口大小
            assert len(input_keypoints) == window_size, f"固定窗口模式下输入帧数应为{window_size}，实际为{len(input_keypoints)}"
        else:
            # 滑动窗口模式下可能需要填充
            # 注意：输入关键点应该已经在predict_3d_pose_thread中处理过，确保长度为self.mixste_args.frames
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
        # 预先测量渲染时间开始
        render_start = time.time()
        
        # 坐标转换
        pose_3d_world = camera_to_world(pose_3d.copy(), R=self.rot, t=0)
        pose_3d_world[:, 2] -= np.min(pose_3d_world[:, 2])
        
        # 创建空白图像
        img = np.ones((self.render_size[0], self.render_size[1], 3), dtype=np.uint8) * 255
        
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
        scale = self.render_scale
        center_x, center_y = self.render_center
        
        # 应用旋转矩阵 (简化版)
        theta = np.radians(self.render_azimuth)  # azimuth
        phi = np.radians(self.render_elevation)    # elevation
        
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
        
        # 计算实际更新间隔
        time_since_last_update = time.time() - self.last_3d_update
        
        # 添加性能信息到图像
        mixste_text = f"3D Prediction: {1.0/self.mixste_time:.1f} FPS"
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
        timestamp = f"更新时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        cv2.putText(img, timestamp, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img
        
    def save_results_to_file(self):
        """保存结果到文件"""
        if not self.save_results:
            return
            
        print("正在保存结果到文件...")
        
        # 创建输出目录结构
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'input_2D'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'output_3D'), exist_ok=True)
        if self.visualization_mode:
            os.makedirs(os.path.join(self.output_dir, 'pose2D'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'pose3D'), exist_ok=True)
            
        # 为视频输入创建特定的文件名
        if self.input_type == 'video':
            # 从视频路径中提取文件名作为标识符
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_prefix = f"{video_name}_"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            info_file_path = os.path.join(self.output_dir, f"{video_name}_{timestamp}_info.txt")
            
            # 保存视频信息到文本文件
            with open(info_file_path, 'w') as f:
                f.write(f"视频文件: {self.video_path}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总帧数: {self.frame_count}\n")
                f.write(f"视频时长: {self.video_duration:.2f}秒\n")
                f.write(f"视频帧率: {self.original_video_fps}fps\n")
                f.write(f"分辨率: {self.frame_width}x{self.frame_height}\n")
                f.write(f"处理模式: {'固定窗口批量处理' if self.use_fixed_window else '滑动窗口连续处理'}\n")
                if self.use_fixed_window:
                    f.write(f"窗口大小: {self.window_size}帧\n")
                    f.write(f"处理窗口数: {self.window_count}\n")
                f.write(f"处理帧数: {len(self.all_keypoints)}\n")
        else:
            # 摄像头模式使用时间戳作为标识符
            output_prefix = "camera_"
            
        # 判断使用哪种方式保存结果
        if self.use_fixed_window:
            # 固定窗口批处理模式
            if len(self.all_window_predictions) == 0:
                print("没有可用的预测结果")
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
            output_2d_path = os.path.join(self.output_dir, 'input_2D', f'{output_prefix}input_keypoints_2d.npz')
            np.savez_compressed(output_2d_path, reconstruction=keypoints_hrnet)
            
            # 保存3D姿态
            if len(all_poses_3d) > 0:
                output_3d_path = os.path.join(self.output_dir, 'output_3D', f'{output_prefix}output_keypoints_3d.npz')
                poses_3d_array = np.array(all_poses_3d)
                np.savez_compressed(output_3d_path, reconstruction=poses_3d_array)
                
            print(f"已保存 {len(all_keypoints)} 帧的结果到 {self.output_dir}")
            
        else:
            # 滑动窗口连续处理模式
            if len(self.all_keypoints) == 0:
                print("没有可用的关键点数据")
                return
                
            # 转换2D关键点到HRNet格式
            keypoints_array = np.array(self.all_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
            scores_array = np.array(self.all_scores)[np.newaxis, ...]  # [1, T, 17]
            
            # 转换格式
            keypoints_hrnet = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
            scores_hrnet = convert_yolov11_to_hrnet_scores(scores_array.copy())
            
            # 保存2D关键点
            output_2d_path = os.path.join(self.output_dir, 'input_2D', f'{output_prefix}input_keypoints_2d.npz')
            np.savez_compressed(output_2d_path, reconstruction=keypoints_hrnet)
            
            # 保存3D姿态
            if len(self.all_poses_3d) > 0:
                output_3d_path = os.path.join(self.output_dir, 'output_3D', f'{output_prefix}output_keypoints_3d.npz')
                poses_3d_array = np.array(self.all_poses_3d)
                np.savez_compressed(output_3d_path, reconstruction=poses_3d_array)
                
            print(f"已保存 {len(self.all_keypoints)} 帧的结果到 {self.output_dir}")
            
        # 为视频创建附加信息
        if self.input_type == 'video':
            print(f"视频处理信息已保存到 {info_file_path}")

    def extract_2d_pose_thread(self):
        """2D姿态提取线程"""
        last_update = time.time()
        
        while not stop_event.is_set():
            try:
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = frame_queue.get()
                
                keypoints, scores = self.extract_2d_pose(frame)
                
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
                        vis_2d = self.create_2d_visualization(frame, keypoints_vis)
                    else:
                        # 没有检测到人，只显示原始帧和状态信息
                        fps_text = f"System Frame Rate: {1.0/self.frame_time:.1f} FPS"
                        yolo_text = f"YOLO Processing: {1.0/self.yolo_time:.1f} FPS"
                        status_text = "状态: 未检测到人体"
                        cv2.putText(vis_2d, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, yolo_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    visualization_2d_queue.put(vis_2d)
                    
                # 记录关键点用于保存
                if self.save_results:
                    self.all_keypoints.append(keypoints)
                    self.all_scores.append(scores)
                    
            except Exception as e:
                print(f"2D姿态提取线程出错: {e}")
                import traceback
                traceback.print_exc()
                
    def predict_3d_pose_thread(self):
        """3D姿态预测线程"""
        buffer_keypoints = []
        update_interval = max(0.05, 1.0 / (self.args.min_3d_update_fps * 2.5))  # 大幅减小更新间隔，提高刷新率
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
                        status_text = "状态: 未检测到人体"
                        cv2.putText(blank_3d, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        fps_text = f"3D渲染: {1.0/max(0.001, render_time):.1f} FPS"
                        avg_fps_text = f"平均更新率: {np.mean(fps_history) if fps_history else 0:.1f} FPS"
                        cv2.putText(blank_3d, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(blank_3d, avg_fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        visualization_3d_queue.put(blank_3d)
                
            except Exception as e:
                print(f"3D姿态预测线程出错: {e}")
                import traceback
                traceback.print_exc()
                
    def process_fixed_window_thread(self):
        """固定窗口批量处理线程，负责收集帧并按照固定窗口进行处理"""
        print("固定窗口处理线程已启动")
        
        while not stop_event.is_set():
            try:
                # 关闭检查 - 如果收到关闭信号且处理完所有帧，则退出
                if self.is_shutting_down and len(self.current_window_keypoints) == 0:
                    print("已处理完所有窗口，线程退出")
                    break
                
                # 收集阶段 - 收集足够的关键点形成一个完整窗口
                if len(self.current_window_keypoints) < self.window_size and not self.window_processing:
                    if keypoints_queue.empty():
                        time.sleep(0.01)
                        continue
                    
                    frame, keypoints, scores = keypoints_queue.get()
                    
                    # 检查关键点是否有效
                    is_valid_keypoints = not np.all(keypoints == 0)
                    
                    # 只添加有效的关键点
                    if is_valid_keypoints:
                        self.current_window_frames.append(frame)
                        self.current_window_keypoints.append(keypoints)
                        self.current_window_scores.append(scores)
                        
                        # 更新预热进度
                        if not self.warmup_completed:
                            self.warmup_frames_collected = len(self.current_window_keypoints)
                            
                        # 在可视化模式下创建2D姿态图
                        if self.visualization_mode and not visualization_2d_queue.full():
                            # 转换关键点到HRNet格式
                            keypoints_array = np.array([keypoints])[np.newaxis, ...]
                            keypoints_converted = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
                            keypoints_vis = keypoints_converted[0, 0]
                            
                            # 创建2D可视化
                            vis_2d = self.create_2d_visualization(frame, keypoints_vis)
                            
                            # 添加收集状态信息
                            if not self.warmup_completed:
                                status_text = f"系统预热中: {len(self.current_window_keypoints)}/{self.window_size} 帧"
                                progress = int(len(self.current_window_keypoints) / self.window_size * 100)
                                cv2.putText(vis_2d, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                # 绘制进度条
                                cv2.rectangle(vis_2d, (10, 150), (10 + int(progress * 2), 170), (0, 255, 0), -1)
                                cv2.rectangle(vis_2d, (10, 150), (210, 170), (0, 0, 0), 1)
                            else:
                                window_text = f"收集窗口 #{self.window_count + 1}: {len(self.current_window_keypoints)}/{self.window_size} 帧"
                                cv2.putText(vis_2d, window_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                            visualization_2d_queue.put(vis_2d)
                    
                # 处理阶段 - 在收集了足够的帧后进行3D姿态预测
                if len(self.current_window_keypoints) >= self.window_size and not self.window_processing:
                    print(f"\n开始处理窗口 #{self.window_count + 1} ({self.window_size}帧)")
                    self.window_processing = True
                    process_start_time = time.time()
                    
                    # 使用当前窗口的关键点进行3D预测
                    window_keypoints = np.array(self.current_window_keypoints)
                    window_3d_predictions = []
                    
                    # 使用模型处理整个窗口
                    pose_3d = self.predict_3d_pose(window_keypoints)
                    
                    # 记录这个窗口的3D预测结果及相关信息
                    window_result = {
                        'window_id': self.window_count,
                        'frames': self.current_window_frames,
                        'keypoints': self.current_window_keypoints,
                        'scores': self.current_window_scores,
                        'pose_3d': pose_3d,
                        'start_frame_index': self.total_frames_processed,
                        'processing_time': time.time() - process_start_time
                    }
                    
                    # 添加到全局结果
                    self.all_window_predictions.append(window_result)
                    
                    # 更新计数和状态
                    self.window_count += 1
                    self.total_frames_processed += len(self.current_window_keypoints)
                    self.window_processing_delay = time.time() - process_start_time
                    self.last_window_completed_time = time.time()
                    
                    # 清空当前窗口，准备收集下一个窗口
                    self.current_window_frames = []
                    self.current_window_keypoints = []
                    self.current_window_scores = []
                    
                    # 标记预热完成
                    if not self.warmup_completed:
                        self.warmup_completed = True
                        print("系统预热完成！开始正常处理")
                    
                    # 重置处理标志
                    self.window_processing = False
                    
                    print(f"窗口 #{self.window_count} 处理完成，用时 {self.window_processing_delay:.2f} 秒")
                
                # 显示处理状态
                if self.visualization_mode and not visualization_3d_queue.full() and self.warmup_completed:
                    # 检查是否有处理完成的窗口
                    if len(self.all_window_predictions) > 0:
                        # 确定当前应该显示哪个窗口的哪一帧
                        current_window_index = self.current_display_index // self.window_size
                        
                        # 如果索引超出范围，表示显示最后一个窗口的最后一帧
                        if current_window_index >= len(self.all_window_predictions):
                            current_window_index = len(self.all_window_predictions) - 1
                            frame_in_window = self.window_size - 1
                        else:
                            frame_in_window = self.current_display_index % self.window_size
                        
                        # 获取当前窗口
                        current_window = self.all_window_predictions[current_window_index]
                        
                        # 创建3D可视化
                        if hasattr(current_window, 'cached_vis_3d') and frame_in_window in current_window['cached_vis_3d']:
                            # 使用缓存的3D可视化
                            vis_3d = current_window['cached_vis_3d'][frame_in_window]
                        else:
                            # 创建新的3D可视化
                            try:
                                vis_3d = self.create_3d_visualization_fixed_window(
                                    current_window['pose_3d'],
                                    window_id=current_window['window_id'],
                                    frame_in_window=frame_in_window,
                                    total_frame_index=current_window['start_frame_index'] + frame_in_window
                                )
                                
                                # 缓存结果
                                if not hasattr(current_window, 'cached_vis_3d'):
                                    current_window['cached_vis_3d'] = {}
                                current_window['cached_vis_3d'][frame_in_window] = vis_3d
                            except Exception as e:
                                print(f"创建3D可视化时出错: {e}")
                                vis_3d = np.ones((800, 800, 3), dtype=np.uint8) * 255
                        
                        # 更新显示索引
                        self.current_display_index += 1
                        
                        # 将3D可视化放入队列
                        visualization_3d_queue.put(vis_3d)
                    else:
                        # 没有可用的窗口结果，显示等待信息
                        blank_3d = np.ones((800, 800, 3), dtype=np.uint8) * 255
                        cv2.putText(blank_3d, "等待第一个窗口处理完成...", (10, 400), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        visualization_3d_queue.put(blank_3d)
                
                # 简单帧率控制
                time.sleep(0.001)
                
            except Exception as e:
                print(f"固定窗口处理线程出错: {e}")
                import traceback
                traceback.print_exc()
                
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
        # 调用现有的OpenCV渲染函数
        img = self.create_3d_visualization_opencv(pose_3d)
        
        # 添加窗口处理信息
        delay_text = f"系统延迟: 约 {self.estimated_delay:.1f} 秒"
        window_text = f"窗口 #{window_id+1}, 帧 {frame_in_window+1}/{self.window_size}"
        total_text = f"总帧数: {total_frame_index+1}"
        
        # 添加半透明背景区域，用于窗口信息
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (400, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 添加信息文本
        cv2.putText(img, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, window_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, total_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        
        return img

    def run(self):
        """运行姿态估计pipeline"""
        self.setup_camera()
        
        # 启动工作线程
        threads = []
        threads.append(threading.Thread(target=self.extract_2d_pose_thread))
        
        # 选择使用哪种处理模式
        if self.use_fixed_window:
            threads.append(threading.Thread(target=self.process_fixed_window_thread))
            print("使用固定窗口批量处理模式")
        else:
            threads.append(threading.Thread(target=self.predict_3d_pose_thread))
            print("使用滑动窗口连续处理模式")
            
        for thread in threads:
            thread.daemon = True
            thread.start()
            
        try:
            print("开始实时姿态估计...")
            print("按 'q' 键退出，按 'p' 键暂停/继续，按 'r' 键重置视频到开始位置")
            
            if self.use_fixed_window:
                print("\n固定窗口批量处理配置:")
                print(f"- 窗口大小: {self.window_size} 帧")
                print(f"- 预计延迟: 约 {self.estimated_delay} 秒")
                print(f"- 处理方式: 每收集{self.window_size}帧进行一次批量处理")
                print("- 初次使用需要预热过程，请耐心等待首个窗口处理完成\n")
            else:
                print("\n分离的处理线程：")
                print("- 主循环：负责帧读取与显示")
                print("- 2D提取线程：运行YOLO模型进行人体检测和关键点提取")
                print("- 3D预测线程：运行MixSTE模型进行3D姿态重建\n")

            # 主循环
            current_frame = 0  # 当前处理的帧数
            last_frame_time = time.time()  # 上一帧的处理时间
            frame_timer = 0  # 帧计时器，用于控制播放速度
            end_of_video = False  # 视频是否播放完毕标志
            
            while True:
                start_time = time.time()
                
                # 帧率控制
                if self.fps_limit > 0:
                    time_since_last_frame = time.time() - self.last_frame_time
                    target_time_between_frames = 1.0 / self.fps_limit
                    
                    # 计算需要等待的时间以维持目标帧率
                    if time_since_last_frame < target_time_between_frames:
                        time.sleep(target_time_between_frames - time_since_last_frame)
                
                # 处理用户输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户请求退出")
                    break
                elif key == ord('p'):
                    # 暂停/继续视频播放
                    self.pause_video = not self.pause_video
                    pause_status = "暂停" if self.pause_video else "继续"
                    print(f"\n视频播放已{pause_status}")
                    continue
                elif key == ord('r') and self.input_type == 'video':
                    # 重置视频到开始位置
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frame = 0
                    end_of_video = False
                    print("\n视频已重置到开始位置")
                    continue
                
                # 如果暂停，跳过帧处理
                if self.pause_video:
                    # 显示上次处理的结果
                    if self.visualization_mode:
                        # 添加暂停提示
                        if not visualization_2d_queue.empty():
                            vis_2d = visualization_2d_queue.get()
                            cv2.putText(vis_2d, "已暂停", (self.frame_width - 150, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow("2D Pose Detection", vis_2d)
                            visualization_2d_queue.put(vis_2d)  # 放回队列
                            
                        if not visualization_3d_queue.empty():
                            vis_3d = visualization_3d_queue.get()
                            cv2.putText(vis_3d, "已暂停", (self.render_size[0] - 150, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow("3D Pose Reconstruction", vis_3d)
                            visualization_3d_queue.put(vis_3d)  # 放回队列
                    
                    time.sleep(0.05)  # 减少CPU使用率
                    continue
                
                # 检查视频播放状态
                if self.input_type == 'video' and end_of_video:
                    if self.video_loop:
                        # 循环播放，重置视频
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        current_frame = 0
                        end_of_video = False
                        print("\n视频已重新开始播放")
                    else:
                        # 非循环模式，显示最后一帧并等待
                        time.sleep(0.1)
                        continue
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    if self.input_type == 'camera':
                        print("无法从摄像头读取帧")
                        break
                    else:
                        # 视频播放完毕
                        end_of_video = True
                        if self.video_loop:
                            continue  # 返回循环开始处重置视频
                        else:
                            print("\n视频播放完毕")
                            if self.visualization_mode:
                                # 显示播放完毕提示
                                if not visualization_2d_queue.empty():
                                    vis_2d = visualization_2d_queue.get()
                                    cv2.putText(vis_2d, "视频播放完毕", (self.frame_width - 250, 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    cv2.imshow("2D Pose Detection", vis_2d)
                                
                            # 等待用户按键
                            key = cv2.waitKey(1000) & 0xFF
                            if key == ord('q'):
                                break
                            elif key == ord('r'):
                                # 重置视频
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                current_frame = 0
                                end_of_video = False
                                continue
                            else:
                                # 继续等待
                                time.sleep(0.1)
                                continue
                
                # 更新帧计数和时间
                current_frame += 1
                self.last_frame_time = time.time()
                self.frame_counter += 1
                
                # 添加视频播放信息到帧
                if self.input_type == 'video':
                    # 计算当前视频时间和进度
                    current_time = current_frame / self.original_video_fps
                    progress = current_frame / self.frame_count * 100
                    
                    # 在帧上显示播放信息
                    time_text = f"时间: {current_time:.2f}s / {self.video_duration:.2f}s"
                    progress_text = f"进度: {progress:.1f}%"
                    cv2.putText(frame, time_text, (10, self.frame_height - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, progress_text, (10, self.frame_height - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                # 添加到帧队列
                if not frame_queue.full():
                    frame_queue.put(frame)
                    
                # 显示原始摄像头画面（不包含姿态标识）
                cv2.imshow("Input Feed", frame)
                
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
                    if self.use_fixed_window:
                        if self.input_type == 'video':
                            print(f"\r视频进度: {progress:.1f}% | Display Rate: {fps:.1f} FPS | YOLO: {1.0/max(0.001, self.yolo_time):.1f} FPS | " +
                                  f"Window: {self.window_count} | Frames: {self.total_frames_processed}", end="")
                        else:
                            print(f"\rDisplay Rate: {fps:.1f} FPS | YOLO: {1.0/max(0.001, self.yolo_time):.1f} FPS | " +
                                  f"Window: {self.window_count} | Frames: {self.total_frames_processed}", end="")
                    else:
                        if self.input_type == 'video':
                            print(f"\r视频进度: {progress:.1f}% | Display Rate: {fps:.1f} FPS | YOLO: {1.0/max(0.001, self.yolo_time):.1f} FPS | " +
                                  f"3D Prediction: {1.0/max(0.001, self.mixste_time):.1f} FPS", end="")
                        else:
                            print(f"\rDisplay Rate: {fps:.1f} FPS | YOLO: {1.0/max(0.001, self.yolo_time):.1f} FPS | " +
                                  f"3D Prediction: {1.0/max(0.001, self.mixste_time):.1f} FPS", end="")
                    
                # 如果是视频且需要保持原始帧率，可能需要额外的等待
                if self.input_type == 'video' and self.fps_limit == self.original_video_fps:
                    elapsed = time.time() - start_time
                    frame_time = 1.0 / self.original_video_fps
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                
            # 处理剩余帧和关闭前的操作
            if self.use_fixed_window and len(self.current_window_keypoints) > 0:
                print(f"\n等待处理剩余的 {len(self.current_window_keypoints)} 帧...")
                self.is_shutting_down = True
                
                # 如果有部分收集的帧但不足一个窗口，填充剩余帧
                if 0 < len(self.current_window_keypoints) < self.window_size:
                    frames_needed = self.window_size - len(self.current_window_keypoints)
                    print(f"填充 {frames_needed} 帧以完成最后一个窗口")
                    
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
                        print("所有窗口处理完成，正在关闭程序...")
                        break
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n用户中断程序")
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
    parser = argparse.ArgumentParser(description='实时姿态估计')
    
    # 输入源参数
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    input_group.add_argument('--video_path', type=str, default=None, help='输入视频文件路径')
    
    # 视频播放控制
    parser.add_argument('--video_loop', action='store_true', help='循环播放视频')
    parser.add_argument('--video_start_time', type=float, default=0, help='视频开始时间(秒)')
    
    # 基本参数
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    
    # 模型参数
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO模型路径')
    parser.add_argument('--mixste_model', type=str, default='checkpoint/pretrained/hot_mixste/model.pth', 
                       help='MixSTE模型路径')
    
    # 可视化参数
    parser.add_argument('--visualization', action='store_true', help='启用可视化模式')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴范围')
    parser.add_argument('--use_opencv_render', action='store_true', help='使用OpenCV进行快速3D渲染')
    parser.add_argument('--min_3d_update_fps', type=float, default=15.0, 
                       help='3D姿态更新的最小目标帧率，更新间隔会相应调整')
    
    # 性能参数
    parser.add_argument('--min_frames_to_start', type=int, default=10,
                       help='开始3D预测所需的最小帧数')
    
    # 其他参数
    parser.add_argument('--conf_thresh', type=float, default=0.5, 
                       help='关键点置信度阈值，低于此值的关键点将使用上一帧的对应关键点')
    parser.add_argument('--save_results', action='store_true', help='保存结果到文件')
    parser.add_argument('--video_fps_limit', type=int, default=30,
                       help='限制处理帧率，避免系统过载，设为0表示不限制')
    
    # 固定窗口批量处理相关参数
    parser.add_argument('--use_fixed_window', action='store_true', help='启用固定窗口批量处理')
    
    return parser.parse_args()
    

def main():
    """主函数"""
    args = parse_args()
    estimator = PoseEstimator(args)
    estimator.run()
    

if __name__ == "__main__":
    main() 