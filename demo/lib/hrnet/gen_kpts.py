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


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False, detector='yolo11', batch_size=200, hrnet_cfg=None):
    """
    生成视频的关键点
    采用批量处理模式：先用YOLO处理所有帧，再用HRNet处理
    Args:
        video: 视频路径
        det_dim: YOLO检测分辨率
        num_peroson: 检测的人数
        gen_output: 是否生成输出
        detector: 使用的检测器类型
        batch_size: 批处理大小
        hrnet_cfg: HRNet配置文件路径，如果为None则使用默认配置
    """
    # 更新配置
    args = get_default_args()
    args.det_dim = det_dim
    args.num_person = num_peroson
    args.video = video
    
    # 使用自定义配置文件
    if hrnet_cfg:
        args.cfg = hrnet_cfg
    
    reset_config(args)

    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算需要处理的批次数
    num_batches = (video_length + batch_size - 1) // batch_size

    # 加载模型
    from lib.yolo11.human_detector import load_model as yolo_model
    from lib.yolo11.human_detector import yolo_human_det as yolo_det
    from lib.yolo11.human_detector import reset_target  # 导入重置目标函数
    has_quiet_param = True
    yolo_batch_size = 16  # YOLO11的批处理大小
        
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)
    
    # 重置目标锁定状态
    reset_target()

    # 用于存储所有结果
    all_keypoints = []
    all_scores = []
    
    # 用于跟踪目标ID
    target_id = None

    # 创建CUDA事件用于异步处理
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    for batch_idx in range(num_batches):
        start_frame = batch_idx * batch_size
        end_frame = min((batch_idx + 1) * batch_size, video_length)
        current_batch_size = end_frame - start_frame
        
        print(f'\n处理第 {batch_idx + 1}/{num_batches} 批 (帧 {start_frame} - {end_frame})')
        
        # 一次性读取所有帧
        frames = []
        for _ in tqdm(range(current_batch_size), desc=f'批次 {batch_idx + 1} - 读取帧'):
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
        
        print('第一阶段: YOLO人体检测...')
        batch_bboxs = []
        batch_scores = []
        
        # 使用YOLO11的批处理功能
        for i in range(0, len(frames), yolo_batch_size):
            batch = frames[i:i + yolo_batch_size]
            bboxs_batch, scores_batch = yolo_det(batch, human_model, reso=det_dim, confidence=args.thred_score, quiet=True)
            batch_bboxs.extend(bboxs_batch)
            batch_scores.extend(scores_batch)
        
        # 重置视频读取位置到当前批次开始
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print('第二阶段: HRNet姿态估计...')
        batch_kpts = []
        batch_kpts_scores = []
        
        # 预分配GPU内存并预热模型
        if torch.cuda.is_available():
            max_batch = min(32, current_batch_size)  # 最大HRNet批处理大小
            dummy_input = torch.zeros((max_batch, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])).cuda()
            _ = pose_model(dummy_input)  # 预热GPU
            del dummy_input
            torch.cuda.empty_cache()
        
        # 收集一批次的输入数据
        batch_inputs = []
        batch_centers = []
        batch_scales = []
        batch_frame_indices = []
        max_batch_size = 32  # HRNet的最大批处理大小
        
        for frame_idx in tqdm(range(len(frames)), desc=f'批次 {batch_idx + 1}'):
            frame = frames[frame_idx]
            
            # 获取当前帧的检测结果
            bboxs = batch_bboxs[frame_idx]
            if bboxs is None:
                continue
                
            # 处理检测框...（保持原有的检测框处理逻辑不变）
            track_bboxs = []
            for bbox in bboxs:
                bbox = [round(i, 2) for i in list(bbox)]
                track_bboxs.append(bbox)

            # 准备HRNet输入
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)
            inputs = inputs[:, [2, 1, 0]]
            
            # 收集批处理数据
            batch_inputs.append(inputs)
            batch_centers.append(center)
            batch_scales.append(scale)
            batch_frame_indices.append(frame_idx)
            
            # 当收集够一个批次或是最后一帧时，进行批处理
            if len(batch_inputs) == max_batch_size or frame_idx == len(frames) - 1:
                # 合并批次数据
                inputs_batch = torch.cat(batch_inputs, dim=0)
                centers_batch = np.concatenate(batch_centers, axis=0)
                scales_batch = np.concatenate(batch_scales, axis=0)
                
                # HRNet处理
                with torch.no_grad():
                    if torch.cuda.is_available():
                        inputs_batch = inputs_batch.cuda(non_blocking=True)
                        start_event.record()
                    
                    output_batch = pose_model(inputs_batch)
                    
                    if torch.cuda.is_available():
                        end_event.record()
                        # 不要立即同步，让GPU继续工作
                    
                    # 计算坐标
                    preds_batch, maxvals_batch = get_final_preds(
                        cfg, 
                        output_batch.cpu().numpy(), 
                        centers_batch,
                        scales_batch
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 现在同步
                
                # 处理每一帧的结果
                for i in range(len(batch_frame_indices)):
                    kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
                    scores = np.zeros((num_peroson, 17), dtype=np.float32)
                    
                    # 只取当前帧的预测结果
                    frame_preds = preds_batch[i:i+1]
                    frame_maxvals = maxvals_batch[i:i+1]
                    
                    for j, kpt in enumerate(frame_preds):
                        kpts[j] = kpt
                    for j, score in enumerate(frame_maxvals):
                        scores[j] = score.squeeze()
                    
                    # 将结果插入正确的位置
                    frame_idx = batch_frame_indices[i]
                    while len(batch_kpts) <= frame_idx:
                        batch_kpts.append(None)
                        batch_kpts_scores.append(None)
                    batch_kpts[frame_idx] = kpts
                    batch_kpts_scores[frame_idx] = scores
                
                # 清空批次数据
                batch_inputs = []
                batch_centers = []
                batch_scales = []
                batch_frame_indices = []
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    del inputs_batch, output_batch
                    torch.cuda.empty_cache()
        
        # 移除空帧
        batch_kpts = [k for k in batch_kpts if k is not None]
        batch_kpts_scores = [s for s in batch_kpts_scores if s is not None]
        
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
