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
        self.use_opencv_render = args.use_opencv_render
        
        # 添加线程同步事件
        self.models_ready = threading.Event()
        self.threads_ready = threading.Event()
        self.first_frame_processed = threading.Event()
        
        # 打印配置信息
        print(f"\n=== 系统配置 ===")
        print(f"设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"摄像头ID: {args.camera_id}")
        print(f"可视化模式: {'启用' if self.visualization_mode else '禁用'}")
        print(f"3D渲染方式: {'OpenCV (快速)' if self.use_opencv_render else 'Matplotlib (详细)'}")
        print(f"处理方案: 固定243帧序列")
        print(f"预期延迟: ~9秒 (30FPS)")
        print(f"结果保存: {'启用' if self.save_results else '禁用'}")
        if self.save_results:
            print(f"输出目录: {self.output_dir}")
        print(f"帧率限制: {self.fps_limit} FPS")
        print(f"==============\n")
        
        # 创建输出目录
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'input_2D'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'output_3D'), exist_ok=True)
            if self.visualization_mode:
                os.makedirs(os.path.join(self.output_dir, 'pose2D'), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, 'pose3D'), exist_ok=True)
        
        # 摄像头设置
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.source_fps = 0
        
        # 用于存储结果的数组
        self.all_keypoints = []
        self.all_scores = []
        self.all_poses_3d = []
        
        # 性能统计
        self.yolo_time = 0
        self.mixste_time = 0
        self.frame_time = 0
        self.last_frame_time = time.time()
        self.last_sequence_end_time = time.time()
        self.last_sequence_prediction_time = 0  # 添加新变量存储最新的序列处理时间
        
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
        print("Loading YOLO model...")
        try:
            self.yolo_model = YOLO(self.args.yolo_model)
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
            
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
            return True
        except Exception as e:
            print(f"Error loading MixSTE model: {e}")
            return False
            
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
        """设置摄像头输入源"""
        print(f"初始化摄像头 (ID: {self.args.camera_id})...")
        self.cap = cv2.VideoCapture(self.args.camera_id)
        
        # 检查是否成功打开
        if not self.cap.isOpened():
            print("无法打开摄像头")
            sys.exit(1)
            
        # 获取摄像头信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_camera_fps = self.source_fps > 0  # 标记是否使用摄像头报告的帧率
        
        # 如果无法获取摄像头帧率，设置默认值
        if not self.is_camera_fps:
            self.source_fps = 30.0  # 默认30fps
            print(f"警告: 无法从摄像头获取帧率，使用默认值 {self.source_fps} FPS")
        
        print(f"摄像头初始化完成:")
        print(f"- 分辨率: {self.frame_width}x{self.frame_height}")
        print(f"- 输入帧率: {self.source_fps} FPS {'(摄像头报告)' if self.is_camera_fps else '(默认值)'}")
        print(f"- 输出帧率: {self.fps_limit} FPS")
        
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
            if len(self.all_keypoints) > 0 and len(self.all_keypoints[-1]) == 17:
                # 对于低置信度的关键点，使用上一帧的对应关键点
                low_conf_mask = scores < self.args.conf_thresh
                if np.any(low_conf_mask):
                    kpts_xy[low_conf_mask] = self.all_keypoints[-1][low_conf_mask]
                    scores[low_conf_mask] = self.all_scores[-1][low_conf_mask]
                
            # 更新最后一次有效检测的时间
            self.last_valid_detection_time = time.time()
        else:
            # 如果没有检测到人体
            if not hasattr(self, 'last_valid_detection_time'):
                self.last_valid_detection_time = 0
            
            # 计算自上次有效检测的时间
            time_since_last_detection = time.time() - self.last_valid_detection_time
            
            # 如果在3秒内丢失检测，使用上一帧的关键点
            if len(self.all_keypoints) > 0 and len(self.all_keypoints[-1]) == 17 and time_since_last_detection < 3.0:
                kpts_xy = self.all_keypoints[-1].copy()
                scores = self.all_scores[-1].copy()
            else:
                # 如果丢失检测超过3秒或没有历史关键点，返回零数组
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
            output_3d_sequence: 3D姿态预测结果序列，形状为[F, 17, 3]
        """
        start_time = time.time()
        
        # 注意：输入关键点应该已经在predict_3d_pose_thread中处理过，确保长度为self.mixste_args.frames
        assert len(input_keypoints) == self.mixste_args.frames, f"Input frame count should be {self.mixste_args.frames}, actual is {len(input_keypoints)}"
        
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
            
        # 获取完整序列的3D姿态
        output_3d_sequence = output_3D[0].cpu().detach().numpy()  # [F, 17, 3]
        output_3d_sequence[:, 0, :] = 0  # 将所有帧的根节点置零
        
        self.mixste_time = time.time() - start_time
        return output_3d_sequence
        
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
        
        # 添加半透明背景以提高文字可读性
        overlay = vis_2d.copy()
        cv2.rectangle(overlay, (5, 5), (300, 140), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, vis_2d, 0.3, 0, vis_2d)
        
        # 添加性能信息
        input_fps_text = f"Input Video: {self.source_fps:.1f} FPS"
        yolo_fps = f"YOLO Process: {1.0/max(0.001, self.yolo_time):.1f} FPS"
        resolution = f"Resolution: {self.frame_width}x{self.frame_height}"
        
        cv2.putText(vis_2d, input_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_2d, yolo_fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_2d, resolution, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
        # 帧率控制 - 更严格的实现
        current_time = time.time()
        if not hasattr(self, 'last_render_time'):
            self.last_render_time = current_time
        else:
            time_since_last_render = current_time - self.last_render_time
            target_render_time = 1.0 / 30.0  # 限制渲染帧率为30FPS
            
            if time_since_last_render < target_render_time:
                # 如果距离上次渲染时间太短，直接返回None
                return None
            
        # 更新最后渲染时间
        self.last_render_time = current_time
        
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
            # 简化的3D到2D投影，翻转Z轴方向
            x = point[0] * np.cos(theta) - point[1] * np.sin(theta)
            # 注意这里改变了y的计算方式，将point[2]前的符号改为负号
            y = -point[2] * np.cos(phi) - (point[0] * np.sin(theta) + point[1] * np.cos(theta)) * np.sin(phi)
            
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
        
        # Z轴 - 注意这里也需要翻转Z轴方向
        z_axis = (int(origin[0]), int(origin[1] - axis_length * np.cos(phi)))  # 改为减号
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
        
        # 添加半透明背景，使文字更容易阅读
        overlay = img.copy()
        cv2.rectangle(overlay, (5, img.shape[0]-145), (350, img.shape[0]-5), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 添加性能信息到图像
        mixste_text = f"3D Prediction: {1.0/self.mixste_time:.1f} FPS"
        render_text = f"Render Method: OpenCV ({current_render_time*1000:.1f}ms)"
        render_fps_text = f"Pure Render: {1.0/max(0.001, current_render_time):.1f} FPS"
        
        # 添加时间戳
        timestamp = f"Last Update: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
        
        # 添加文字
        cv2.putText(img, mixste_text, (10, img.shape[0] - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, render_text, (10, img.shape[0] - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, render_fps_text, (10, img.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, timestamp, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img
        
    def save_results_to_file(self):
        """保存结果到文件"""
        if not self.save_results or len(self.all_keypoints) == 0:
            return
            
        print("\nSaving results to file...")
        
        try:
            # 过滤出有效的关键点数据
            valid_keypoints = []
            valid_scores = []
            valid_poses_3d = []
            
            for i, (keypoints, scores) in enumerate(zip(self.all_keypoints, self.all_scores)):
                # 检查关键点数组的形状和有效性
                if isinstance(keypoints, np.ndarray) and keypoints.shape == (17, 2):
                    valid_keypoints.append(keypoints)
                    valid_scores.append(scores)
                    if i < len(self.all_poses_3d):
                        valid_poses_3d.append(self.all_poses_3d[i])
            
            if len(valid_keypoints) == 0:
                print("No valid keypoint data to save")
                return
                
            print(f"Processed {len(self.all_keypoints)} frames, valid frames: {len(valid_keypoints)}")
            
            # 转换2D关键点到HRNet格式
            keypoints_array = np.array(valid_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
            scores_array = np.array(valid_scores)[np.newaxis, ...]  # [1, T, 17]
            
            # 转换格式
            keypoints_hrnet = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
            scores_hrnet = convert_yolov11_to_hrnet_scores(scores_array.copy())
            
            # 保存2D关键点
            output_2d_path = os.path.join(self.output_dir, 'input_2D', 'input_keypoints_2d.npz')
            np.savez_compressed(output_2d_path, reconstruction=keypoints_hrnet)
            print(f"2D keypoint data saved to: {output_2d_path}")
            
            # 保存3D姿态
            if len(valid_poses_3d) > 0:
                output_3d_path = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
                poses_3d_array = np.array(valid_poses_3d)
                np.savez_compressed(output_3d_path, reconstruction=poses_3d_array)
                print(f"3D pose data saved to: {output_3d_path}")
                
            print(f"All results saved to directory: {self.output_dir}")
            
        except Exception as e:
            print(f"Error occurred while saving results: {e}")
            import traceback
            traceback.print_exc()

    def extract_2d_pose_thread(self):
        """2D姿态提取线程"""
        last_update = time.time()
        
        # 新添加：通知主线程该线程已准备就绪
        self.threads_ready.set()
        
        while not stop_event.is_set():
            try:
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame, processing_complete = frame_queue.get()  # 修改：获取事件对象
                
                # 更新处理时间
                current_time = time.time()
                self.frame_time = current_time - last_update
                last_update = current_time
                
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
                        # 添加半透明背景以提高文字可读性
                        overlay = vis_2d.copy()
                        cv2.rectangle(overlay, (5, 5), (300, 140), (255, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.7, vis_2d, 0.3, 0, vis_2d)
                        
                        # 添加性能信息
                        input_fps_text = f"Input Video: {self.source_fps:.1f} FPS"
                        yolo_fps = f"YOLO Process: {1.0/max(0.001, self.yolo_time):.1f} FPS"
                        status_text = "Status: No Person Detected"
                        resolution = f"Resolution: {self.frame_width}x{self.frame_height}"
                        
                        cv2.putText(vis_2d, input_fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, yolo_fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis_2d, resolution, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                    visualization_2d_queue.put(vis_2d)
                    
                    # 新添加：通知主线程第一帧已处理完成
                    if not self.first_frame_processed.is_set():
                        self.first_frame_processed.set()
                    
                # 记录关键点用于保存
                if self.save_results:
                    self.all_keypoints.append(keypoints)
                    self.all_scores.append(scores)
                    
                # 通知主线程当前帧处理完成
                processing_complete.set()

            except Exception as e:
                print(f"2D Pose Extraction Thread Error: {e}")
                import traceback
                traceback.print_exc()
                
    def predict_3d_pose_thread(self):
        """3D姿态预测线程 - 使用固定243帧序列处理方案"""
        buffer_keypoints = []  # 用于存储关键点序列
        is_warmup = True  # 预热标志
        sequence_counter = 0  # 序列计数器
        
        # 序列播放管理
        current_sequence = None  # 当前播放的序列
        next_sequence = None    # 下一个待播放的序列
        current_frame_idx = 0   # 当前播放帧索引
        
        # 用于存储所有3D序列的列表
        all_3d_sequences = []  # 存储所有243帧序列
        
        # 性能监控变量
        prediction_times = []  # 记录预测时间
        last_sequence_end_time = time.time()  # 上一个序列结束时间
        last_frame_update = time.time()  # 上一帧更新时间
        target_frame_time = 1.0 / 30.0  # 目标帧间隔时间(30FPS)
        
        # 人体检测状态
        no_person_counter = 0  # 计数器：连续多少帧没有检测到人
        last_valid_keypoints = None  # 最后一个有效的关键点
        sequence_started = False  # 标记是否已经开始收集序列
        
        print("\n=== 等待检测到人体... ===")
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                
                # 处理新的关键点数据
                if not keypoints_queue.empty():
                    frame, keypoints, scores = keypoints_queue.get()
                    
                    # 检查关键点是否有效
                    is_valid_keypoints = not np.all(keypoints == 0)
                    
                    if is_valid_keypoints:
                        # 重置无人计数器
                        no_person_counter = 0
                        # 更新最后一个有效关键点
                        last_valid_keypoints = keypoints.copy()
                        
                        # 如果还没开始收集序列，现在开始
                        if not sequence_started:
                            sequence_started = True
                            is_warmup = True
                            print("\n=== 检测到人体，开始收集序列 ===")
                            print("收集初始243帧...")
                            print("预计预热时间: 8-10秒\n")
                            
                        # 添加到缓冲区
                        buffer_keypoints.append(keypoints)
                        
                        # 在预热阶段显示进度
                        if is_warmup and len(buffer_keypoints) % 30 == 0:
                            progress = (len(buffer_keypoints) / 243) * 100
                            print(f"\r预热进度: {progress:.1f}% ({len(buffer_keypoints)}/243 帧)", end="")
                    else:
                        # 增加无人计数器
                        no_person_counter += 1
                        
                        if sequence_started:
                            # 如果序列已经开始，处理无人情况
                            if no_person_counter >= 90:  # 3秒无人
                                if len(buffer_keypoints) < 50:
                                    # 如果缓冲区帧数较少，清空并重置
                                    print("\n\n=== 长时间未检测到人体，重置序列 ===")
                                    buffer_keypoints = []
                                    sequence_started = False
                                    is_warmup = True
                                    last_valid_keypoints = None
                                    current_sequence = None
                                    next_sequence = None
                                    continue
                                elif len(buffer_keypoints) >= 243:
                                    # 如果已经收集了完整序列，停止继续处理
                                    print("\n\n=== 序列完成，未检测到人体，等待新的人体出现 ===")
                                    buffer_keypoints = []
                                    sequence_started = False
                                    is_warmup = True
                                    last_valid_keypoints = None
                                    continue
                            
                            # 使用最后的有效关键点填充
                            if last_valid_keypoints is not None and len(buffer_keypoints) > 0:
                                buffer_keypoints.append(last_valid_keypoints.copy())
                                if len(buffer_keypoints) % 30 == 0:
                                    print(f"\r使用最后的有效关键点填充序列 ({len(buffer_keypoints)}/243 帧)", end="")
                    
                    # 当积累了243帧时进行处理
                    if sequence_started and len(buffer_keypoints) >= 243:
                        sequence_start_time = time.time()
                        
                        # 准备输入数据
                        input_keypoints = np.array(buffer_keypoints[:243])
                        
                        # 预测3D姿态序列
                        new_sequence = self.predict_3d_pose(input_keypoints)
                        
                        # 保存完整的243帧序列
                        all_3d_sequences.append(new_sequence)
                        
                        # 更新序列管理
                        if current_sequence is None:
                            current_sequence = new_sequence
                            current_frame_idx = 0
                        else:
                            next_sequence = new_sequence
                        
                        # 计算处理时间
                        prediction_time = time.time() - sequence_start_time
                        prediction_times.append(prediction_time)
                        
                        # 如果是预热阶段
                        if is_warmup:
                            print("\n\n=== 预热完成 ===")
                            print(f"首次预测时间: {prediction_time:.2f} 秒")
                            print("开始正常预测流程...\n")
                            is_warmup = False
                        else:
                            # 计算序列间隔时间
                            sequence_interval = time.time() - last_sequence_end_time
                            avg_prediction_time = np.mean(prediction_times)
                            print(f"\r序列 #{sequence_counter} 完成 "
                                  f"(处理时间: {prediction_time:.2f}秒, "
                                  f"平均: {avg_prediction_time:.2f}秒, "
                                  f"间隔: {sequence_interval:.2f}秒)", end="")
                            
                            # 更新最新的序列处理时间
                            self.last_sequence_prediction_time = prediction_time
                        
                        # 清空缓冲区，准备下一个序列
                        buffer_keypoints = buffer_keypoints[243:]
                        self.last_sequence_end_time = time.time()
                        sequence_counter += 1
                
                # 处理3D可视化更新
                if current_sequence is not None and self.visualization_mode:
                    # 控制帧率
                    target_frame_time = 1.0 / self.fps_limit if self.fps_limit > 0 else 1.0 / 30.0
                    if current_time - last_frame_update >= target_frame_time:
                        # 创建3D可视化
                        vis_3d = self.create_3d_visualization(current_sequence[current_frame_idx])
                        
                        # 只有当渲染成功时才更新显示
                        if vis_3d is not None:
                            # 添加序列播放信息
                            cv2.putText(vis_3d, f"Sequence #{sequence_counter}", (10, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(vis_3d, f"Frame {current_frame_idx + 1}/243", (10, 120), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # 清空可视化队列并放入新帧
                            with visualization_3d_queue.mutex:
                                visualization_3d_queue.queue.clear()
                            visualization_3d_queue.put(vis_3d)
                            
                            # 更新帧索引和序列
                            current_frame_idx += 1
                            if current_frame_idx >= 243:
                                if next_sequence is not None:
                                    # 如果有下一个序列准备好了，就切换到下一个序列
                                    current_sequence = next_sequence
                                    next_sequence = None
                                    current_frame_idx = 0
                                else:
                                    # 如果下一个序列还没准备好，就停在当前序列的最后一帧
                                    current_frame_idx = 242
                            
                            last_frame_update = current_time
                
                elif self.visualization_mode:
                    # 显示等待状态
                    blank_3d = np.ones((800, 800, 3), dtype=np.uint8) * 255
                    if not sequence_started:
                        status_text = "Waiting for person detection..."
                        cv2.putText(blank_3d, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif is_warmup:
                        status_text = f"System warming up... ({len(buffer_keypoints)}/243 frames)"
                        frames_remaining = 243 - len(buffer_keypoints)
                        time_remaining = f"Estimated time remaining: {frames_remaining/30:.1f}s"
                        cv2.putText(blank_3d, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if len(buffer_keypoints) > 0:
                            status_text = f"Processing next sequence... ({len(buffer_keypoints)}/243 frames)"
                            frames_remaining = 243 - len(buffer_keypoints)
                            time_remaining = f"Estimated time remaining: {frames_remaining/30:.1f}s"
                            cv2.putText(blank_3d, status_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(blank_3d, time_remaining, (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(blank_3d, "Waiting for next sequence...", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                        cv2.putText(blank_3d, f"Completed sequences: {sequence_counter}", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 放入队列
                    with visualization_3d_queue.mutex:
                        visualization_3d_queue.queue.clear()
                    visualization_3d_queue.put(blank_3d)
                
                # 计算实际帧率
                self.frame_time = time.time() - current_time
                
                # 显示性能信息
                if self.frame_counter % 30 == 0:  # 每30帧更新一次控制台输出
                    # 使用最新的序列处理时间
                    sequence_time_str = f"{self.last_sequence_prediction_time:.2f}秒" if self.last_sequence_prediction_time > 0 else "等待序列完成"
                    
                    # 添加帧率来源标识
                    fps_source = "(摄像头)" if self.is_camera_fps else "(默认)"
                    
                    print(f"\r输入帧率: {self.source_fps:.1f} FPS {fps_source} | "
                          f"YOLO处理: {1.0/max(0.001, self.yolo_time):.1f} FPS | "
                          f"上一序列处理时间: {sequence_time_str}", end="")
                
                # 短暂休眠以减少CPU使用
                time.sleep(0.001)
                
            except Exception as e:
                print(f"\n3D姿态预测线程错误: {e}")
                import traceback
                traceback.print_exc()
                
        # 处理程序退出时的剩余帧
        if sequence_started and len(buffer_keypoints) > 0:
            print("\n\n=== 处理剩余帧 ===")
            print(f"剩余帧数: {len(buffer_keypoints)}")
            
            # 如果剩余帧数不足243，通过复制最后一帧来填充
            if len(buffer_keypoints) < 243:
                pad_length = 243 - len(buffer_keypoints)
                print(f"填充 {pad_length} 帧以完成最后一个序列")
                buffer_keypoints.extend([buffer_keypoints[-1]] * pad_length)
            
            # 处理最后一个序列
            input_keypoints = np.array(buffer_keypoints[:243])
            final_sequence = self.predict_3d_pose(input_keypoints)
            
            # 保存最后的序列
            all_3d_sequences.append(final_sequence)
            sequence_counter += 1
            
            print("所有帧处理完成")
            
        # 保存完整的3D序列数据
        if len(all_3d_sequences) > 0:
            # 将所有序列连接成一个大的数组
            all_sequences_array = np.concatenate(all_3d_sequences, axis=0)
            
            # 保存完整序列
            if self.save_results:
                output_3d_path = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
                np.savez_compressed(output_3d_path, 
                                  reconstruction=all_sequences_array,
                                  sequence_length=243,
                                  total_sequences=len(all_3d_sequences))
                print(f"\n保存了 {len(all_3d_sequences)} 个序列，"
                      f"总帧数: {len(all_3d_sequences) * 243}")
                print(f"3D姿态数据已保存到: {output_3d_path}")

    def initialize_models(self):
        """初始化所有模型并等待它们准备就绪"""
        print("\n=== 正在初始化模型 ===")
        
        # 初始化YOLO模型
        if not self._init_yolo_model():
            return False
            
        # 初始化MixSTE模型
        if not self._init_mixste_model():
            return False
            
        print("\n所有模型初始化完成！")
        self.models_ready.set()
        return True

    def run(self):
        """运行姿态估计pipeline"""
        # 首先初始化模型
        if not self.initialize_models():
            print("模型初始化失败，程序退出")
            return
            
        # 等待模型准备就绪
        print("\n等待模型准备就绪...")
        self.models_ready.wait()
        
        # 设置摄像头
        self.setup_camera()
        
        # 启动工作线程
        threads = []
        threads.append(threading.Thread(target=self.extract_2d_pose_thread))
        threads.append(threading.Thread(target=self.predict_3d_pose_thread))
            
        for thread in threads:
            thread.daemon = True
            thread.start()
            
        # 等待所有线程准备就绪
        print("\n等待处理线程准备就绪...")
        self.threads_ready.wait()
            
        try:
            print("\n开始实时姿态估计...")
            print("按 'q' 退出, 'p' 暂停/继续")
            print("\n处理线程:")
            print("- 主循环: 帧捕获和显示")
            print("- 2D线程: YOLO模型人体检测和关键点提取")
            print("- 3D线程: MixSTE模型3D姿态重建\n")

            # 播放控制变量
            paused = False
            processing_complete = threading.Event()
            current_time = time.time()
            
            # 主循环
            while True:
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户请求退出")
                    break
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        print("\n已暂停")
                    else:
                        print("\n继续运行")
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("\n无法从摄像头读取帧")
                    break
                
                # 更新当前时间
                current_time = time.time()
                self.frame_counter += 1
                
                # 重置处理完成事件
                processing_complete.clear()
                
                # 如果是第一帧，等待2D检测线程准备就绪
                if self.frame_counter == 1:
                    print("\n等待第一帧处理完成...")
                    frame_queue.put((frame, processing_complete))
                    self.first_frame_processed.wait()
                    print("第一帧处理完成，开始正常运行...")
                else:
                    # 添加到帧队列并等待处理完成
                    if not frame_queue.full():
                        frame_queue.put((frame, processing_complete))
                        processing_complete.wait()
                
                # 显示原始画面
                cv2.imshow("Input Video", frame)
                
                # 显示2D可视化结果
                if self.visualization_mode and not visualization_2d_queue.empty():
                    vis_2d = visualization_2d_queue.get()
                    cv2.imshow("2D Pose Detection", vis_2d)
                
                # 显示3D可视化结果
                if self.visualization_mode:
                    if not visualization_3d_queue.empty():
                        vis_3d = visualization_3d_queue.get()
                        cv2.imshow("3D Pose Reconstruction", vis_3d)
                    elif self.frame_counter % 30 == 0:
                        blank_3d = np.ones((800, 800, 3), dtype=np.uint8) * 255
                        status_text = "Status: Waiting for next 243-frame sequence..."
                        cv2.putText(blank_3d, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("3D Pose Reconstruction", blank_3d)
                
                # 计算实际帧率
                self.frame_time = time.time() - current_time
                
                # 显示性能信息
                if self.frame_counter % 30 == 0:  # 每30帧更新一次控制台输出
                    # 使用最新的序列处理时间
                    sequence_time_str = f"{self.last_sequence_prediction_time:.2f}秒" if self.last_sequence_prediction_time > 0 else "等待序列完成"
                    
                    # 添加帧率来源标识
                    fps_source = "(摄像头)" if self.is_camera_fps else "(默认)"
                    
                    print(f"\r输入帧率: {self.source_fps:.1f} FPS {fps_source} | "
                          f"YOLO处理: {1.0/max(0.001, self.yolo_time):.1f} FPS | "
                          f"上一序列处理时间: {sequence_time_str}", end="")
                
                # 短暂休眠以减少CPU使用
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
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
    parser = argparse.ArgumentParser(description='实时姿态估计 - 固定243帧序列处理')
    
    # 输入参数
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头设备ID')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output', help='结果保存目录')
    
    # 模型参数
    parser.add_argument('--yolo_model', type=str, default='yolo11m-pose.pt', help='YOLO模型路径')
    parser.add_argument('--mixste_model', type=str, default='checkpoint/pretrained/hot_mixste/model.pth', 
                       help='MixSTE模型路径')
    
    # 可视化参数
    parser.add_argument('--visualization', action='store_true', help='启用可视化模式')
    parser.add_argument('--fix_z', action='store_true', help='固定3D可视化中的Z轴范围')
    parser.add_argument('--use_opencv_render', action='store_true', help='使用OpenCV进行快速3D渲染')
    
    # 其他参数
    parser.add_argument('--conf_thresh', type=float, default=0.5, 
                       help='关键点置信度阈值，低于此值将使用上一帧的关键点')
    
    args = parser.parse_args()
    
    # 设置默认行为
    args.save_results = True  # 始终保存结果到文件
    args.video_fps_limit = 30  # 摄像头输入固定30fps
    
    return args
    
def main():
    """主函数"""
    args = parse_args()
    estimator = PoseEstimator(args)
    estimator.run()

if __name__ == "__main__":
    main() 