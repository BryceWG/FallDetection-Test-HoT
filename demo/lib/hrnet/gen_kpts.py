from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds
from lib.hrnet.lib.models import pose_hrnet

cfg_dir = 'demo/lib/hrnet/experiments/'
model_dir = 'demo/lib/checkpoint/'

# Loading human detector model
from lib.yolov3.human_detector import load_model as yolo_model
from lib.yolov3.human_detector import yolo_human_det as yolo_det
from lib.sort.sort import Sort


def get_default_args():
    """
    返回默认参数而不是解析命令行
    """
    class Args:
        def __init__(self):
            self.cfg = cfg_dir + 'w48_384x288_adam_lr1e-3.yaml'
            self.modelDir = model_dir + 'pose_hrnet_w48_384x288.pth'
            self.det_dim = 416
            self.thred_score = 0.30
            self.animation = False
            self.num_person = 1
            self.video = 'camera'
            self.gpu = '0'
            self.fix_z = False
            self.opts = []
    
    return Args()


def parse_args():
    """
    返回默认参数而不是解析命令行
    """
    return get_default_args()


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')
    
    return model


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False, detector='yolov3', batch_size=200):
    """
    生成视频的关键点
    采用批量处理模式：先用YOLO处理所有帧，再用HRNet处理
    Args:
        batch_size
    """
    # 更新配置
    args = get_default_args()
    args.det_dim = det_dim
    args.num_person = num_peroson
    args.video = video
    reset_config(args)

    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算需要处理的批次数
    num_batches = (video_length + batch_size - 1) // batch_size

    # 加载模型
    if detector == 'yolo11':
        from lib.yolo11.human_detector import load_model as yolo_model
        from lib.yolo11.human_detector import yolo_human_det as yolo_det
        from lib.yolo11.human_detector import reset_target  # 导入重置目标函数
        has_quiet_param = True
        yolo_batch_size = 16  # YOLO11的批处理大小
    else:  # 默认使用YOLOv3
        from lib.yolov3.human_detector import load_model as yolo_model
        from lib.yolov3.human_detector import yolo_human_det as yolo_det
        has_quiet_param = False
        yolo_batch_size = 1
        
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)
    
    # 重置目标锁定状态（如果使用YOLO11）
    if detector == 'yolo11':
        reset_target()

    # 用于存储所有结果
    all_keypoints = []
    all_scores = []
    
    # 用于跟踪目标ID
    target_id = None

    for batch_idx in range(num_batches):
        start_frame = batch_idx * batch_size
        end_frame = min((batch_idx + 1) * batch_size, video_length)
        current_batch_size = end_frame - start_frame
        
        print(f'\n处理第 {batch_idx + 1}/{num_batches} 批 (帧 {start_frame} - {end_frame})')
        
        print('第一阶段: YOLO人体检测...')
        # 第一阶段：YOLO处理当前批次的所有帧
        batch_bboxs = []
        batch_scores = []
        
        if detector == 'yolo11':
            # 使用YOLO11的批处理功能
            frames = []
            frame_count = 0
            
            for _ in tqdm(range(current_batch_size), desc=f'批次 {batch_idx + 1} - 读取帧'):
                ret, frame = cap.read()
                if not ret:
                    continue
                frames.append(frame)
                frame_count += 1
                
                # 当收集够一个YOLO批次或是最后一帧时，进行批处理
                if len(frames) == yolo_batch_size or frame_count == current_batch_size:
                    bboxs_batch, scores_batch = yolo_det(frames, human_model, reso=det_dim, confidence=args.thred_score, quiet=True)
                    
                    # 处理每一帧的结果
                    for i, (bboxs, scores) in enumerate(zip(bboxs_batch, scores_batch)):
                        batch_bboxs.append(bboxs)
                        batch_scores.append(scores)
                    
                    frames = []  # 清空帧缓存
                    
        else:
            # 原始的逐帧处理方式
            for _ in tqdm(range(current_batch_size), desc=f'批次 {batch_idx + 1}'):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)
                batch_bboxs.append(bboxs)
                batch_scores.append(scores)

        # 重置视频读取位置到当前批次开始
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print('第二阶段: HRNet姿态估计...')
        # 第二阶段：HRNet处理当前批次
        batch_kpts = []
        batch_kpts_scores = []
        
        for frame_idx in tqdm(range(current_batch_size), desc=f'批次 {batch_idx + 1}'):
            ret, frame = cap.read()
            if not ret:
                continue
                
            # 获取当前帧的检测结果
            bboxs = batch_bboxs[frame_idx]
            if bboxs is None:
                continue
            
            # 如果使用YOLO11，我们已经在YOLO阶段锁定了目标，直接使用检测结果
            if detector == 'yolo11':
                track_bboxs = []
                for bbox in bboxs:
                    bbox = [round(i, 2) for i in list(bbox)]
                    track_bboxs.append(bbox)
            else:
                # 使用Sort跟踪器更新
                people_track = people_sort.update(bboxs)
                
                # 如果是第一帧且有检测结果，锁定目标ID
                if batch_idx == 0 and frame_idx == 0 and people_track.shape[0] > 0:
                    # 计算每个检测框的面积
                    areas = []
                    for i in range(people_track.shape[0]):
                        bbox = people_track[i, :-1]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        areas.append((i, area, people_track[i, -1]))  # (索引, 面积, ID)
                    
                    # 按面积降序排序
                    areas.sort(key=lambda x: x[1], reverse=True)
                    
                    # 锁定面积最大的人体ID
                    target_id = areas[0][2]
                    print(f"已锁定目标ID: {target_id}, 面积: {areas[0][1]:.2f}")
                
                # 如果已锁定目标ID，只保留该ID的检测框
                if target_id is not None:
                    target_idx = -1
                    for i in range(people_track.shape[0]):
                        if people_track[i, -1] == target_id:
                            target_idx = i
                            break
                    
                    if target_idx >= 0:
                        people_track_ = people_track[target_idx, :-1].reshape(1, 4)
                    else:
                        # 如果当前帧未找到目标ID，使用上一帧的结果
                        if batch_kpts:
                            # 跳过当前帧，使用上一帧的关键点
                            batch_kpts.append(batch_kpts[-1])
                            batch_kpts_scores.append(batch_kpts_scores[-1])
                            continue
                        else:
                            # 如果没有上一帧结果，尝试使用当前帧的第一个检测结果
                            if people_track.shape[0] > 0:
                                people_track_ = people_track[0, :-1].reshape(1, 4)
                            else:
                                continue
                else:
                    # 如果未锁定目标ID，使用Sort的默认行为
                    if people_track.shape[0] == 1:
                        people_track_ = people_track[-1, :-1].reshape(1, 4)
                    elif people_track.shape[0] >= 2:
                        people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
                        people_track_ = people_track_[::-1]
                    else:
                        continue

                track_bboxs = []
                for bbox in people_track_:
                    bbox = [round(i, 2) for i in list(bbox)]
                    track_bboxs.append(bbox)

            # HRNet处理
            with torch.no_grad():
                inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)
                inputs = inputs[:, [2, 1, 0]]
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                output = pose_model(inputs)
                
                # 计算坐标
                preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

            # 保存结果
            kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
            scores = np.zeros((num_peroson, 17), dtype=np.float32)
            
            for i, kpt in enumerate(preds):
                kpts[i] = kpt
            for i, score in enumerate(maxvals):
                scores[i] = score.squeeze()

            batch_kpts.append(kpts)
            batch_kpts_scores.append(scores)
        
        # 将当前批次的结果添加到总结果中
        all_keypoints.extend(batch_kpts)
        all_scores.extend(batch_kpts_scores)

    # 转换结果格式
    keypoints = np.array(all_keypoints)
    scores = np.array(all_scores)
    
    # 添加错误处理和调试信息
    print(f"关键点数组形状: {keypoints.shape}")
    
    # 检查数组是否为空
    if len(keypoints) == 0:
        print("警告: 未检测到任何关键点，返回空数组")
        # 返回符合预期形状的空数组
        return np.zeros((1, 0, 17, 2)), np.zeros((1, 0, 17))
    
    # 检查维度是否足够进行转置
    if len(keypoints.shape) < 4:
        print(f"警告: 关键点数组维度不足 ({len(keypoints.shape)}D)，无法进行转置操作")
        # 如果只有一帧，添加时间维度
        if len(keypoints.shape) == 3:  # (M, N, 2)
            keypoints = np.expand_dims(keypoints, axis=0)  # (1, M, N, 2)
            scores = np.expand_dims(scores, axis=0)  # (1, M, N)
            keypoints = keypoints.transpose(1, 0, 2, 3)  # (M, 1, N, 2)
            scores = scores.transpose(1, 0, 2)  # (M, 1, N)
        else:
            # 返回符合预期形状的空数组
            return np.zeros((1, 0, 17, 2)), np.zeros((1, 0, 17))
    else:
        keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
        scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
