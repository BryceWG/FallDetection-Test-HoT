# python vis_yolov11.py --video sample_video.mp4 --yolo_model yolo11m-pose.pt --stream_buffer
import os
import sys
import cv2
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import warnings
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from model.mixste.hot_mixste import Model

# 导入自定义模块
from data_processor import normalize_screen_coordinates, convert_yolov11_to_hrnet_keypoints, convert_yolov11_to_hrnet_scores

def get_pose2D_yolov11(video_path, output_dir, model_path, conf_thresh=0.5, batch_size=4, stream_buffer=False):

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")
    
    # 加载YOLO v11 Pose模型
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(model_path).to(device)
    except Exception as e:
        raise Exception(f"加载YOLO模型失败: {str(e)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")
        
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_length == 0:
        raise Exception("视频文件为空")
    
    print('\n使用YOLO v11 Pose生成2D姿态...')
    
    # 初始化关键点和分数数组
    all_keypoints = []
    all_scores = []
    
    # 设置批处理大小
    frames_batch = []
    frame_indices = []  # 记录每个批次中帧的索引
    
    print(f'使用设备: {device}, 批处理大小: {batch_size}, {"启用" if stream_buffer else "禁用"}流缓冲')
    
    # 使用tqdm显示处理进度
    with tqdm(total=video_length) as pbar:
        # 使用stream=True进行内存优化
        for frame_idx in range(video_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_batch.append(frame)
            frame_indices.append(frame_idx)
            
            # 当收集到一个批次的帧或到达视频末尾时进行处理
            if len(frames_batch) == batch_size or frame_idx == video_length - 1:
                try:
                    # 使用stream=True进行批处理
                    results = model(frames_batch, 
                                 conf=conf_thresh,  # 使用置信度阈值
                                 stream=True,  # 启用stream模式
                                 device=device,
                                 stream_buffer=stream_buffer,  # 控制流缓冲
                                 verbose=False)  # 关闭冗余输出
                    
                    # 处理每一帧的结果
                    for result, curr_frame_idx in zip(results, frame_indices):
                        if result.keypoints is not None and len(result.keypoints.data) > 0:
                            # 获取所有检测到的人的关键点
                            all_detected_keypoints = result.keypoints.data.cpu().numpy()
                            
                            # 如果检测到多个人,选择关键点置信度最高的那个
                            if len(all_detected_keypoints) > 1:
                                mean_conf_scores = np.mean(all_detected_keypoints[:, :, 2], axis=1)
                                best_person_idx = np.argmax(mean_conf_scores)
                                keypoints = all_detected_keypoints[best_person_idx]
                            else:
                                keypoints = all_detected_keypoints[0]
                            
                            kpts_xy = keypoints[:, :2]
                            scores = keypoints[:, 2]
                            
                            # 处理低置信度关键点
                            if len(all_keypoints) > 0:
                                low_conf_mask = scores < conf_thresh
                                kpts_xy[low_conf_mask] = all_keypoints[-1][low_conf_mask]
                                scores[low_conf_mask] = all_scores[-1][low_conf_mask]
                            
                            all_keypoints.append(kpts_xy)
                            all_scores.append(scores)
                        else:
                            # 如果没有检测到人，使用上一帧的关键点或零填充
                            if len(all_keypoints) > 0:
                                all_keypoints.append(all_keypoints[-1])
                                all_scores.append(all_scores[-1])
                            else:
                                all_keypoints.append(np.zeros((17, 2)))
                                all_scores.append(np.zeros(17))
                                
                except Exception as e:
                    print(f"\n批处理帧 {frame_indices[0]} 到 {frame_indices[-1]} 失败: {str(e)}")
                    # 对于失败的批次,填充零数据
                    for _ in frame_indices:
                        if len(all_keypoints) > 0:
                            all_keypoints.append(all_keypoints[-1])
                            all_scores.append(all_scores[-1])
                        else:
                            all_keypoints.append(np.zeros((17, 2)))
                            all_scores.append(np.zeros(17))
                
                # 更新进度条
                pbar.update(len(frames_batch))
                
                # 清空批处理列表
                frames_batch = []
                frame_indices = []
    
    cap.release()
    
    # 检查是否有足够的有效检测
    frames_with_detection = sum(1 for scores in all_scores if np.mean(scores) > 0)
    
    if frames_with_detection == 0:
        raise Exception("视频中未检测到任何人体姿态，请确保视频中有清晰可见的人物")
    
    print(f"\n处理了 {video_length} 帧，其中 {frames_with_detection} 帧成功检测到人体姿态")
    
    if frames_with_detection < video_length * 0.1:  # 如果少于10%的帧检测到人体
        print("警告：大部分帧都未能检测到人体姿态，这可能会影响结果质量")
    
    # 转换为需要的格式 [1, T, 17, 2] 和 [1, T, 17]
    keypoints_array = np.array(all_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
    scores_array = np.array(all_scores)[np.newaxis, ...]  # [1, T, 17]
    
    # 将YOLOv11 Pose的关键点转换为HRNet格式
    keypoints_array = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
    scores_array = convert_yolov11_to_hrnet_scores(scores_array.copy())
    
    print('使用YOLO v11 Pose生成2D姿态成功!')
    
    # 保存关键点
    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)
    output_npz = output_dir + 'input_keypoints_2d.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints_array)
    
    return keypoints_array, scores_array


def get_pose3D(video_path, output_dir, fix_z):

    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.token_num, args.layer_index = 81, 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    args.n_joints, args.out_joints = 17, 17

    ## 加载模型
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # 预训练模型应放在 'checkpoint/pretrained/hot_mixste'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.eval()

    ## 加载输入数据
    keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_chunks = video_length // args.frames + 1
    offset = (n_chunks * args.frames - video_length) // 2

    ret, frame = cap.read()
    if not ret:
        raise Exception("无法读取视频第一帧")
    img_size = frame.shape

    ## 生成3D姿态
    print('\n生成3D姿态...')

    for i in tqdm(range(n_chunks)):

        ## 输入帧
        start_index = i*args.frames - offset
        end_index = (i+1)*args.frames - offset

        low_index = max(start_index, 0)
        high_index = min(end_index, video_length)
        pad_left = low_index - start_index
        pad_right = end_index - high_index

        if pad_left != 0 or pad_right != 0:
            input_2D_no = np.pad(keypoints[0][low_index:high_index], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
        else:
            input_2D_no = keypoints[0][low_index:high_index]
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## 估计3D姿态
        with torch.no_grad():
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if pad_left != 0 and pad_right != 0:
            output_3D = output_3D[:, pad_left:-pad_right]
            input_2D_no = input_2D_no[pad_left:-pad_right]
        elif pad_left != 0:
            output_3D = output_3D[:, pad_left:]
            input_2D_no = input_2D_no[pad_left:]
        elif pad_right != 0:
            output_3D = output_3D[:, :-pad_right]
            input_2D_no = input_2D_no[:-pad_right]

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0].cpu().detach().numpy()

        if i == 0:
            output_3d_all = post_out
        else:
            output_3d_all = np.concatenate([output_3d_all, post_out], axis = 0)

    ## 保存3D关键点
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('生成3D姿态成功!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='输入视频')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴')
    parser.add_argument('--yolo_model', type=str, default='yolo11m-pose.pt', help='YOLO v11 Pose模型路径')
    parser.add_argument('--conf_thresh', type=float, default=0, help='关键点置信度阈值,低于此值的关键点将使用上一帧的对应关键点')
    parser.add_argument('--batch_size', type=int, default=6, help='批处理大小,根据显存大小调整,设置为1禁用批处理')
    parser.add_argument('--stream_buffer', action='store_true', help='是否启用流缓冲,启用后可确保不丢帧,但可能增加延迟')

    args = parser.parse_args()

    video_path = args.video
    if not video_path.startswith(('/', './', '../')):  # 如果不是绝对路径或相对路径
        video_path = os.path.join('./demo/video', video_path)  # 添加默认路径
        
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join('./demo/output', video_name + '/')
    
    # 使用YOLO v11 Pose模型获取2D姿态
    get_pose2D_yolov11(
        video_path, 
        output_dir, 
        args.yolo_model, 
        args.conf_thresh,
        args.batch_size,
        args.stream_buffer
    )
    
    # 使用现有的3D姿态重建模型
    get_pose3D(video_path, output_dir, args.fix_z)
    
    print('3D姿态数据已保存到', output_dir + 'output_3D/output_keypoints_3d.npz')

if __name__ == "__main__":
    main() 