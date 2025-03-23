import os 
import sys
import cv2
import glob
import copy
import json
import torch
import argparse
import numpy as np
import time
import datetime
import queue
import threading
from collections import deque
from tqdm import tqdm
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import warnings
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model

class Buffer:
    """缓冲区类，用于在不同处理阶段之间传递数据"""
    def __init__(self, maxsize=300):
        self.queue = queue.Queue(maxsize)
        self.lock = threading.Lock()
    
    def put(self, item, block=True, timeout=None):
        with self.lock:
            return self.queue.put(item, block=block, timeout=timeout)
    
    def get(self, block=True, timeout=None):
        with self.lock:
            return self.queue.get(block=block, timeout=timeout)
    
    def empty(self):
        with self.lock:
            return self.queue.empty()
    
    def full(self):
        with self.lock:
            return self.queue.full()
    
    def qsize(self):
        with self.lock:
            return self.queue.qsize()

human_tracking = {
    "current_id": None,      # 当前跟踪的人体ID
    "frames_without_human": 0,  # 连续没有检测到人体的帧数
    "reset_required": False,  # 是否需要重置数据流
    "status_message": "System initializing"  # 状态信息
}

def update_status(message):
    """
    更新状态消息，确保使用ASCII字符
    """
    human_tracking["status_message"] = message

def load_models(args):
    """
    加载所有需要的模型
    """
    print("Loading models...")
    
    # 加载人体检测器
    if args.detector == 'yolo11':
        from lib.yolo11.human_detector import load_model as yolo_model
        from lib.yolo11.human_detector import reset_target
        from lib.yolo11.human_detector import get_default_args as get_yolo_args
        
        # 使用默认参数加载YOLO模型
        yolo_args = get_yolo_args()
        reset_target()
        detector = yolo_model(yolo_args, inp_dim=832)
        print("Loaded YOLO model")
    else:  # 默认使用YOLOv3
        from lib.yolov3.human_detector import load_model as yolo_model
        detector = yolo_model(inp_dim=832)  # 使用与原代码相同的参数
        print("Loaded YOLOv3 model")
    
    # 加载姿态估计模型 (HRNet)
    from lib.hrnet.lib.config import cfg, update_config
    from lib.hrnet.lib.models.pose_hrnet import PoseHighResolutionNet
    from lib.hrnet.gen_kpts import parse_args
    
    # 使用与原代码相同的配置
    hrnet_cfg = 'demo/lib/hrnet/experiments/w48_512x384_adam_lr1e-3.yaml'
    
    hrnet_args = parse_args()
    hrnet_args.cfg = hrnet_cfg
    update_config(cfg, hrnet_args)
    
    pose_model = PoseHighResolutionNet(cfg)
    pose_model.load_state_dict(torch.load(cfg.OUTPUT_DIR))
    pose_model = pose_model.cuda()
    pose_model.eval()
    print("Loaded HRNet model")
    
    # 加载3D姿态估计模型 (MixSTE)
    mixste_args = copy.deepcopy(args)
    mixste_args.layers, mixste_args.channel, mixste_args.d_hid, mixste_args.frames = 8, 512, 1024, 243
    mixste_args.token_num, mixste_args.layer_index = 81, 3
    mixste_args.pad = (mixste_args.frames - 1) // 2
    mixste_args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    mixste_args.n_joints, mixste_args.out_joints = 17, 17
    
    mixste_model = Model(mixste_args).cuda()
    model_dict = mixste_model.state_dict()
    model_path = sorted(glob.glob(os.path.join(mixste_args.previous_dir, '*.pth')))[0]
    pre_dict = torch.load(model_path)
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    mixste_model.load_state_dict(model_dict)
    mixste_model.eval()
    print("Loaded MixSTE model")
    
    print("Loaded all models")
    return detector, pose_model, mixste_model

def detect_human(frame, detector, args):
    """
    使用YOLO检测人体
    """
    try:
        if args.detector == 'yolo11':
            from lib.yolo11.human_detector import yolo_human_det as yolo_det
            bboxes, status = yolo_det(frame, detector, args.detector)
            if status is not None:
                # 更新状态消息
                update_status(status)
            return bboxes
        else:
            from lib.yolov3.human_detector import yolo_human_det as yolo_det
            bboxes = yolo_det(frame, detector)
            return bboxes
    except Exception as e:
        update_status("Detection error")
        return None

def extract_pose2D(frame, bboxes, pose_model):
    """
    使用HRNet提取2D姿态
    """
    from lib.hrnet.lib.utils.inference import get_max_preds
    from lib.hrnet.lib.config import cfg
    
    # 初始化返回值
    keypoints = np.zeros((1, 17, 2))
    scores = np.zeros((1, 17))
    
    # 检查边界框是否为None或空列表
    if bboxes is None or len(bboxes) == 0:
        return None, None
    
    try:
        # 预处理
        with torch.no_grad():
            inputs = torch.zeros(1, 3, 384, 288).cuda()
            valid_bbox_found = False
            
            for i, bbox in enumerate(bboxes):
                if i >= 1:  # 只处理第一个检测到的人
                    break
                    
                # 确保边界框坐标是有效的
                if bbox is None or len(bbox) < 4:
                    continue
                    
                # 确保边界框坐标是整数并且在图像范围内
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(frame.shape[1], int(bbox[2]))
                y2 = min(frame.shape[0], int(bbox[3]))
                
                # 确保边界框有足够的大小
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                    
                # 裁剪并调整大小
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                
                valid_bbox_found = True
                    
                person_img = cv2.resize(person_img, (288, 384))
                person_img = person_img[:, :, ::-1]  # BGR to RGB
                
                person_img = person_img / 255.
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                person_img = (person_img - mean) / std
                person_img = person_img.transpose((2, 0, 1))
                
                inputs[0] = torch.from_numpy(person_img).float().cuda()
                
                # 前向传播
                output = pose_model(inputs)
                
                # 获取关键点
                preds, _ = get_max_preds(output.detach().cpu().numpy())
                
                # 缩放回原始大小
                preds[0, :, 0] = preds[0, :, 0] * (bbox[2] - bbox[0]) / 288 + bbox[0]
                preds[0, :, 1] = preds[0, :, 1] * (bbox[3] - bbox[1]) / 384 + bbox[1]
                
                # 计算置信度
                heatmaps = output.detach().cpu().numpy()
                scores = np.zeros((1, preds.shape[1]))
                for n in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        hm = heatmaps[n][j]
                        scores[n, j] = np.max(hm)
                
                return preds[0], scores[0]
            
            # 如果没有找到有效的边界框
            if not valid_bbox_found:
                return None, None
    
    except Exception as e:
        return None, None
    
    return None, None

def predict_pose3D(keypoints_2d_buffer, mixste_model, args, img_size):
    """
    使用MixSTE预测3D姿态
    """
    # 检查输入数据是否有效
    if keypoints_2d_buffer is None or len(keypoints_2d_buffer) == 0:
        return None
    
    # 确保输入数据是numpy数组
    if not isinstance(keypoints_2d_buffer, np.ndarray):
        keypoints_2d_buffer = np.array(keypoints_2d_buffer)
        
    # 检查关键点数据的形状
    if not all(kp.shape == (17, 2) for kp in keypoints_2d_buffer):
        return None
    
    # 转换为输入格式
    with torch.no_grad():
        frames = len(keypoints_2d_buffer)
        input_2D_no = keypoints_2d_buffer
        
        # 规范化坐标
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  
        
        # 数据增强（左右翻转）
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        
        try:
            input_2D_aug = copy.deepcopy(input_2D)
            input_2D_aug[:, :, 0] *= -1
            input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
            
            input_2D = input_2D[np.newaxis, :, :, :, :]
            input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
            
            # 前向传播
            predicted_3d_pos = mixste_model(input_2D)
            
            # 取出结果
            predicted_3d_pos = predicted_3d_pos[0, 0].cpu().numpy()
            
            # 返回3D姿态
            return predicted_3d_pos
        except Exception as e:
            return None

def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='实时3D姿态估计')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--detector', type=str, default='yolo11', choices=['yolov3', 'yolo11'], help='选择人体检测器')
    parser.add_argument('--output_file', type=str, default='realtime_3d_pose.npy', help='3D姿态输出文件名')
    
    # 解析已知参数,忽略未知参数(这些参数可能是HRNet的)
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 加载模型
    detector, pose_model, mixste_model = load_models(args)
    
    # 创建缓冲区
    yolo_buffer = Buffer(maxsize=15)  # 约0.5s @ 30fps
    hrnet_buffer = Buffer(maxsize=30)  # 约1s @ 30fps
    mixste_buffer = deque(maxlen=243)  # MixSTE的输入
    
    # 存储3D关键点
    poses_3d = []
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_size = (height, width, 3)
    
    # 帧率统计
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    
    # 姿态提取线程
    def pose_extraction_thread():
        while True:
            if yolo_buffer.empty():
                time.sleep(0.01)
                continue
                
            frame = yolo_buffer.get()
            bboxes = detect_human(frame, detector, args)
            
            # 确保bboxes不为None
            if bboxes is None:
                bboxes = []
            
            # 过滤掉无效的边界框
            valid_bboxes = []
            for bbox in bboxes:
                if bbox is not None and len(bbox) == 4:
                    # 确保边界框坐标有效
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        valid_bboxes.append(bbox)
            
            # 处理人体跟踪状态
            if len(valid_bboxes) > 0:
                # 找到最大的边界框（假设是主要目标）
                max_area = 0
                max_bbox = None
                for bbox in valid_bboxes:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_bbox = bbox
                
                # 计算边界框中心点作为简单的ID标识
                if max_bbox is not None:
                    center_x = (max_bbox[0] + max_bbox[2]) / 2
                    center_y = (max_bbox[1] + max_bbox[3]) / 2
                    current_id = f"{int(center_x)}_{int(center_y)}"
                    
                    # 检查ID是否变化
                    if human_tracking["current_id"] is None:
                        # 首次检测到人体
                        human_tracking["current_id"] = current_id
                        human_tracking["frames_without_human"] = 0
                        human_tracking["reset_required"] = True
                        update_status("Human detected")
                    elif human_tracking["current_id"] != current_id:
                        # ID变化，需要重置
                        human_tracking["current_id"] = current_id
                        human_tracking["frames_without_human"] = 0
                        human_tracking["reset_required"] = True
                        update_status("Human ID changed")
                    else:
                        # 同一个人，继续跟踪
                        human_tracking["frames_without_human"] = 0
                        update_status("Tracking human")
                
                # 提取姿态
                keypoints, scores = extract_pose2D(frame, [max_bbox], pose_model)
                if keypoints is not None:
                    hrnet_buffer.put((frame, keypoints, scores, human_tracking["reset_required"]))
                    # 重置标志已处理
                    human_tracking["reset_required"] = False
            else:
                # 没有检测到人
                human_tracking["frames_without_human"] += 1
                if human_tracking["frames_without_human"] > 30:  # 1秒没有检测到人
                    if human_tracking["current_id"] is not None:
                        human_tracking["current_id"] = None
                        human_tracking["reset_required"] = True
                        update_status("No human detected")
                
                # 放回一个空结果
                hrnet_buffer.put((frame, None, None, human_tracking["reset_required"]))
    
    # 3D姿态预测线程
    def pose3d_prediction_thread():
        nonlocal fps_count, fps_display
        
        while True:
            if hrnet_buffer.empty():
                time.sleep(0.01)
                continue
                
            frame, keypoints, scores, reset_required = hrnet_buffer.get()
            
            # 如果需要重置，清空MixSTE缓冲区
            if reset_required and len(mixste_buffer) > 0:
                update_status("Resetting data stream")
                mixste_buffer.clear()
            
            if keypoints is not None:
                # 添加到MixSTE缓冲区
                mixste_buffer.append(keypoints)
                
                # 当收集到至少30帧时，进行一次3D预测
                if len(mixste_buffer) >= 30:
                    # 准备输入数据
                    if len(mixste_buffer) < 243:
                        # 不足243帧，复制最后一帧填充
                        buffer_list = list(mixste_buffer)
                        while len(buffer_list) < 243:
                            buffer_list.append(buffer_list[-1])
                        
                        # 使用填充后的数据
                        input_keypoints = buffer_list
                    else:
                        # 已有243帧，直接使用
                        input_keypoints = list(mixste_buffer)
                    
                    # 预测3D姿态
                    pose3d = predict_pose3D(input_keypoints, mixste_model, args, img_size)
                    
                    # 保存结果
                    if pose3d is not None:
                        poses_3d.append(pose3d)  # 保存结果
                        fps_count += 1
                        update_status("3D pose estimated successfully")
            
            # 计算FPS
            if time.time() - fps_time > 1.0:  # 每秒更新一次
                fps_display = fps_count
                fps_count = 0
    
    # 启动线程
    pose_thread = threading.Thread(target=pose_extraction_thread)
    pose_thread.daemon = True
    pose_thread.start()
    
    pose3d_thread = threading.Thread(target=pose3d_prediction_thread)
    pose3d_thread.daemon = True
    pose3d_thread.start()
    
    print("System started, press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get camera frame")
                break
            
            # 将帧放入YOLO缓冲区
            if not yolo_buffer.full():
                yolo_buffer.put(frame)
            
            # 显示当前画面
            current_time = time.time()
            if current_time - fps_time > 1.0:
                fps_time = current_time
            
            # 显示FPS和状态信息
            cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Poses: {len(poses_3d)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 确保状态消息使用UTF-8编码
            status_text = human_tracking.get('status_message', 'Initializing...')
            try:
                # 尝试编码并解码，确保文本可以正确显示
                status_text = status_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                status_text = 'System initializing...'
            
            cv2.putText(frame, f"Status: {status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Realtime 3D Pose Estimation', frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("User interrupted")
    
    finally:
        # 保存3D姿态数据
        if poses_3d:
            poses_3d_array = np.array(poses_3d)
            output_file = args.output_file
            np.save(output_file, poses_3d_array)
            print(f"Saved 3D pose data to {output_file}")
            print(f"Generated {len(poses_3d)} frames of 3D pose data")
            print(f"Average processing speed: {fps_display} FPS")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()