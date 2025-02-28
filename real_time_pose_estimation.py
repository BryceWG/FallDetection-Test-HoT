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
frame_queue = queue.Queue(maxsize=10)
keypoints_queue = queue.Queue(maxsize=10)
pose_3d_queue = queue.Queue(maxsize=10)
visualization_2d_queue = queue.Queue(maxsize=5)  # 分离2D可视化队列
visualization_3d_queue = queue.Queue(maxsize=5)  # 分离3D可视化队列
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
        
        # 打印重要的配置信息
        print(f"\n=== 系统配置 ===")
        print(f"设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"可视化模式: {'启用' if self.visualization_mode else '禁用'}")
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
        
        # 设置3D图形
        if self.visualization_mode:
            self.fig = plt.figure(figsize=(5, 5))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.canvas = FigureCanvasAgg(self.fig)
            
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
        """设置摄像头"""
        print(f"正在初始化摄像头 (ID: {self.args.camera_id})...")
        self.cap = cv2.VideoCapture(self.args.camera_id)
        
        # 检查摄像头是否正常打开
        if not self.cap.isOpened():
            print("无法打开摄像头")
            sys.exit(1)
            
        # 获取摄像头信息，不再手动设置分辨率
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"摄像头初始化成功，分辨率: {self.frame_width}x{self.frame_height}")
        
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
        
        # 注意：输入关键点应该已经在predict_3d_pose_thread中处理过，确保长度为self.mixste_args.frames
        assert len(input_keypoints) == self.mixste_args.frames, f"输入帧数应为{self.mixste_args.frames}，实际为{len(input_keypoints)}"
            
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
        mid_frame = self.mixste_args.frames // 2
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
        
    def save_results_to_file(self):
        """保存结果到文件"""
        if not self.save_results or len(self.all_keypoints) == 0:
            return
            
        print("正在保存结果到文件...")
        
        # 转换2D关键点到HRNet格式
        keypoints_array = np.array(self.all_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
        scores_array = np.array(self.all_scores)[np.newaxis, ...]  # [1, T, 17]
        
        # 转换格式
        keypoints_hrnet = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
        scores_hrnet = convert_yolov11_to_hrnet_scores(scores_array.copy())
        
        # 保存2D关键点
        output_2d_path = os.path.join(self.output_dir, 'input_2D', 'input_keypoints_2d.npz')
        np.savez_compressed(output_2d_path, reconstruction=keypoints_hrnet)
        
        # 保存3D姿态
        if len(self.all_poses_3d) > 0:
            output_3d_path = os.path.join(self.output_dir, 'output_3D', 'output_keypoints_3d.npz')
            poses_3d_array = np.array(self.all_poses_3d)
            np.savez_compressed(output_3d_path, reconstruction=poses_3d_array)
            
        print(f"结果已保存到 {self.output_dir}")

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
                
                # 将结果放入队列
                if not keypoints_queue.full():
                    keypoints_queue.put((frame, keypoints, scores))
                    
                # 创建2D可视化并放入队列
                if self.visualization_mode and not visualization_2d_queue.full():
                    vis_2d = self.create_2d_visualization(frame, keypoints)
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
        update_interval = max(1.0, 1.0 / self.args.min_3d_update_fps)  # 限制更新频率
        last_update = time.time()
        
        while not stop_event.is_set():
            try:
                if keypoints_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame, keypoints, scores = keypoints_queue.get()
                
                # 添加到缓冲区
                buffer_keypoints.append(keypoints)
                
                # 管理缓冲区大小，保留最新的帧
                if len(buffer_keypoints) > self.mixste_args.frames:
                    # 如果缓冲区超过需要的帧数，删除最旧的帧
                    buffer_keypoints = buffer_keypoints[-self.mixste_args.frames:]
                    
                # 控制3D预测的更新频率
                current_time = time.time()
                
                # 确保有足够的帧用于初始预测
                min_frames_needed = min(self.args.min_frames_to_start, self.mixste_args.frames // 8)
                
                if (current_time - last_update >= update_interval and 
                    len(buffer_keypoints) >= min_frames_needed):
                    
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
                            pad_front += np.random.normal(0, 0.5, pad_front.shape)
                            
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
                        pose_3d_queue.put((frame, keypoints, pose_3d))
                        
                    # 创建3D可视化并放入队列
                    if self.visualization_mode and not visualization_3d_queue.full():
                        vis_3d = self.create_3d_visualization(pose_3d)
                        visualization_3d_queue.put(vis_3d)
                        
                    # 记录3D姿态用于保存
                    if self.save_results:
                        self.all_poses_3d.append(pose_3d)
                        
                    last_update = current_time
                    self.last_3d_update = current_time
                    
            except Exception as e:
                print(f"3D姿态预测线程出错: {e}")
                import traceback
                traceback.print_exc()
                
    def run(self):
        """运行姿态估计pipeline"""
        self.setup_camera()
        
        # 启动工作线程
        threads = []
        threads.append(threading.Thread(target=self.extract_2d_pose_thread))
        threads.append(threading.Thread(target=self.predict_3d_pose_thread))
            
        for thread in threads:
            thread.daemon = True
            thread.start()
            
        try:
            print("开始实时姿态估计...")
            print("按 'q' 键退出")
            print("\n分离的处理线程：")
            print("- 主循环：负责帧读取与显示")
            print("- 2D提取线程：运行YOLO模型进行人体检测和关键点提取")
            print("- 3D预测线程：运行MixSTE模型进行3D姿态重建\n")

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
                    print("无法从摄像头读取帧")
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
                    
                # 显示3D可视化结果（分离显示）
                if self.visualization_mode and not visualization_3d_queue.empty():
                    vis_3d = visualization_3d_queue.get()
                    cv2.imshow("3D Pose Reconstruction", vis_3d)
                    
                # 计算帧率
                self.frame_time = time.time() - start_time
                
                # 显示性能信息
                fps = 1.0 / max(0.001, self.frame_time)  # 避免除以零
                if self.frame_counter % 30 == 0:  # 每30帧更新一次控制台输出
                    print(f"\rDisplay Rate: {fps:.1f} FPS | YOLO Processing: {1.0/max(0.001, self.yolo_time):.1f} FPS | 3D Prediction: {1.0/max(0.001, self.mixste_time):.1f} FPS", end="")
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户请求退出")
                    break
                    
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
    
    # 基本参数
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    
    # 模型参数
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO模型路径')
    parser.add_argument('--mixste_model', type=str, default='checkpoint/pretrained/hot_mixste/model.pth', 
                       help='MixSTE模型路径')
    
    # 可视化参数
    parser.add_argument('--visualization', action='store_true', help='启用可视化模式')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴范围')
    parser.add_argument('--min_3d_update_fps', type=float, default=5.0, 
                       help='3D姿态更新的最小帧率，实际更新频率可能更低')
    
    # 性能参数
    parser.add_argument('--min_frames_to_start', type=int, default=30,
                       help='开始3D预测所需的最小帧数')
    
    # 其他参数
    parser.add_argument('--conf_thresh', type=float, default=0.5, 
                       help='关键点置信度阈值，低于此值的关键点将使用上一帧的对应关键点')
    parser.add_argument('--save_results', action='store_true', help='保存结果到文件')
    parser.add_argument('--video_fps_limit', type=int, default=30,
                       help='限制处理帧率，避免系统过载，设为0表示不限制')
    
    return parser.parse_args()
    

def main():
    """主函数"""
    args = parse_args()
    estimator = PoseEstimator(args)
    estimator.run()
    

if __name__ == "__main__":
    main() 