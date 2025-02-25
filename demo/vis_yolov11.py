import os
import sys
import cv2
import glob
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

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


def convert_yolov11_to_hrnet_keypoints(keypoints):
    """
    将YOLOv11 pose关键点格式转换为HRNet关键点格式
    
    参数:
        keypoints: YOLOv11 pose提取的关键点,形状为 [T, 17, 2] 或 [B, T, 17, 2]
        
    返回:
        转换后的关键点,形状与输入相同但符合HRNet格式
    """
    # 确保输入是numpy数组
    keypoints = np.array(keypoints)
    
    # 获取输入形状
    input_shape = keypoints.shape
    
    # 如果是批次数据,需要处理每个批次
    if len(input_shape) == 4:  # [B, T, 17, 2]
        converted_keypoints = np.zeros((input_shape[0], input_shape[1], 17, 2))
        for b in range(input_shape[0]):
            converted_keypoints[b] = convert_yolov11_to_hrnet_keypoints(keypoints[b])
        return converted_keypoints
    
    # 处理单个序列 [T, 17, 2]
    T = input_shape[0]
    converted_keypoints = np.zeros((T, 17, 2))
    
    for t in range(T):
        # 获取当前帧的关键点
        kpts = keypoints[t]
        
        # 创建新的关键点数组
        new_kpts = np.zeros((17, 2))
        
        # YOLOv11关键点索引(减1转为0基索引)
        nose = 0
        left_eye = 1
        right_eye = 2
        left_ear = 3
        right_ear = 4
        left_shoulder = 5
        right_shoulder = 6
        left_elbow = 7
        right_elbow = 8
        left_wrist = 9
        right_wrist = 10
        left_hip = 11
        right_hip = 12
        left_knee = 13
        right_knee = 14
        left_ankle = 15
        right_ankle = 16
        
        # 映射关键点
        # 0: 躯干中心点(Pelvis/Hip center)
        new_kpts[0] = (kpts[left_hip] + kpts[right_hip]) / 2
        
        # 1: 右臀部(Right hip)
        new_kpts[1] = kpts[right_hip]
        
        # 2: 右膝盖(Right knee)
        new_kpts[2] = kpts[right_knee]
        
        # 3: 右脚踝(Right ankle)
        new_kpts[3] = kpts[right_ankle]
        
        # 4: 左臀部(Left hip)
        new_kpts[4] = kpts[left_hip]
        
        # 5: 左膝盖(Left knee)
        new_kpts[5] = kpts[left_knee]
        
        # 6: 左脚踝(Left ankle)
        new_kpts[6] = kpts[left_ankle]
        
        # 计算颈部位置(颈部是左右肩膀的中点)
        neck = (kpts[left_shoulder] + kpts[right_shoulder]) / 2
        
        # 7: 脊柱(Spine/Mid-back) - 改进的估算方法
        # 脊柱位置是髋部中点向颈部方向移动1/3的距离
        hip_center = new_kpts[0]
        spine_vector = neck - hip_center
        new_kpts[7] = hip_center + spine_vector * (1/3)  # 髋部中点向颈部移动1/3的距离
        
        # 8: 颈部(Neck)
        new_kpts[8] = neck
        
        # 9: 头部(Head) - 改进的估算方法
        # 使用左右眼的中心位置
        eyes_center = (kpts[left_eye] + kpts[right_eye]) / 2
        new_kpts[9] = eyes_center
        
        # 10: 头顶(Head top) - 改进的估算方法
        # 基于眼睛中心和颈部位置计算头顶
        head_direction = eyes_center - neck  # 从颈部到眼睛中心的向量
        head_length = np.linalg.norm(head_direction)
        if head_length > 0:
            head_direction = head_direction / head_length  # 单位向量
            new_kpts[10] = eyes_center + head_direction * head_length  # 从眼睛中心延伸与眼睛到颈部相同的距离
        else:
            # 如果颈部和眼睛中心重合,使用垂直方向
            new_kpts[10] = eyes_center + np.array([0, -15])  # 向上15个像素
        
        # 11: 左肩膀(Left shoulder)
        new_kpts[11] = kpts[left_shoulder]
        
        # 12: 左肘部(Left elbow)
        new_kpts[12] = kpts[left_elbow]
        
        # 13: 左手腕(Left wrist)
        new_kpts[13] = kpts[left_wrist]
        
        # 14: 右肩膀(Right shoulder)
        new_kpts[14] = kpts[right_shoulder]
        
        # 15: 右肘部(Right elbow)
        new_kpts[15] = kpts[right_elbow]
        
        # 16: 右手腕(Right wrist)
        new_kpts[16] = kpts[right_wrist]
        
        converted_keypoints[t] = new_kpts
    
    return converted_keypoints


def convert_yolov11_to_hrnet_scores(scores):
    """
    将YOLOv11 pose的关键点置信度转换为HRNet格式
    
    参数:
        scores: YOLOv11 pose关键点的置信度,形状为 [T, 17] 或 [B, T, 17]
        
    返回:
        转换后的置信度,形状与输入相同但符合HRNet格式
    """
    # 确保输入是numpy数组
    scores = np.array(scores)
    
    # 获取输入形状
    input_shape = scores.shape
    
    # 如果是批次数据,需要处理每个批次
    if len(input_shape) == 3:  # [B, T, 17]
        converted_scores = np.zeros((input_shape[0], input_shape[1], 17))
        for b in range(input_shape[0]):
            converted_scores[b] = convert_yolov11_to_hrnet_scores(scores[b])
        return converted_scores
    
    # 处理单个序列 [T, 17]
    T = input_shape[0]
    converted_scores = np.zeros((T, 17))
    
    for t in range(T):
        # 获取当前帧的置信度
        s = scores[t]
        
        # 创建新的置信度数组
        new_scores = np.zeros(17)
        
        # YOLOv11关键点索引(减1转为0基索引)
        nose = 0
        left_eye = 1
        right_eye = 2
        left_ear = 3
        right_ear = 4
        left_shoulder = 5
        right_shoulder = 6
        left_elbow = 7
        right_elbow = 8
        left_wrist = 9
        right_wrist = 10
        left_hip = 11
        right_hip = 12
        left_knee = 13
        right_knee = 14
        left_ankle = 15
        right_ankle = 16
        
        # 映射置信度
        # 0: 躯干中心点(Pelvis/Hip center)
        new_scores[0] = min(s[left_hip], s[right_hip])
        
        # 1: 右臀部(Right hip)
        new_scores[1] = s[right_hip]
        
        # 2: 右膝盖(Right knee)
        new_scores[2] = s[right_knee]
        
        # 3: 右脚踝(Right ankle)
        new_scores[3] = s[right_ankle]
        
        # 4: 左臀部(Left hip)
        new_scores[4] = s[left_hip]
        
        # 5: 左膝盖(Left knee)
        new_scores[5] = s[left_knee]
        
        # 6: 左脚踝(Left ankle)
        new_scores[6] = s[left_ankle]
        
        # 7: 脊柱(Spine/Mid-back)
        neck_conf = min(s[left_shoulder], s[right_shoulder])
        new_scores[7] = min(new_scores[0], neck_conf)
        
        # 8: 颈部(Neck)
        new_scores[8] = neck_conf
        
        # 9: 头部(Head) - 使用鼻子置信度
        new_scores[9] = s[nose]
        
        # 10: 头顶(Head top) - 估算位置的置信度,使用鼻子置信度
        new_scores[10] = s[nose] * 0.8  # 略低于鼻子置信度
        
        # 11: 左肩膀(Left shoulder)
        new_scores[11] = s[left_shoulder]
        
        # 12: 左肘部(Left elbow)
        new_scores[12] = s[left_elbow]
        
        # 13: 左手腕(Left wrist)
        new_scores[13] = s[left_wrist]
        
        # 14: 右肩膀(Right shoulder)
        new_scores[14] = s[right_shoulder]
        
        # 15: 右肘部(Right elbow)
        new_scores[15] = s[right_elbow]
        
        # 16: 右手腕(Right wrist)
        new_scores[16] = s[right_wrist]
        
        converted_scores[t] = new_scores
    
    return converted_scores


def show2Dpose(kps, img):
    # 定义颜色: 绿色(左侧), 黄色(右侧), 蓝色(中线)
    colors = [(138, 201, 38),    # 绿色 - 左侧
              (25, 130, 196),    # 蓝色 - 中线
              (255, 202, 58)]    # 黄色 - 右侧

    # 定义连接关系
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    # 定义左右侧: 1=左侧(绿色), 2=右侧(黄色), 3=中线(蓝色)
    # 根据人体解剖学正确分配颜色:
    # 0-1, 1-2, 2-3: 右腿 - 右侧色
    # 0-4, 4-5, 5-6: 左腿 - 左侧色
    # 0-7, 7-8, 8-9, 9-10: 脊柱和头部 - 中线色
    # 8-11, 11-12, 12-13: 左臂 - 左侧色
    # 8-14, 14-15, 15-16: 右臂 - 右侧色
    LR = [2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 2, 2, 2]

    thickness = 3

    # 首先绘制连接线
    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
    
    # 然后绘制关节点,使用红色增强可见度
    for i in range(len(kps)):
        point = tuple(map(int, kps[i]))
        cv2.circle(img, point, radius=5, color=(0, 0, 255), thickness=-1)

    return img


def show3Dpose(vals, ax, fix_z):
    ax.view_init(elev=15., azim=70)

    # 定义颜色: 绿色(左侧), 黄色(右侧), 蓝色(中线)
    colors = [(138/255, 201/255, 38/255),  # 绿色 - 左侧
              (255/255, 202/255, 58/255),  # 黄色 - 右侧
              (25/255, 130/255, 196/255)]  # 蓝色 - 中线

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    # 定义左右侧: 1=左侧(绿色), 2=右侧(黄色), 3=中线(蓝色)
    # 根据人体解剖学正确分配颜色:
    # 0-1, 0-4: 髋部连接
    # 1-2, 2-3: 右腿
    # 4-5, 5-6: 左腿
    # 0-7, 7-8, 8-9, 9-10: 脊柱和头部
    # 8-11, 11-12, 12-13: 左臂
    # 8-14, 14-15, 15-16: 右臂
    LR = [2, 1, 2, 1, 2, 1, 3, 3, 2, 1, 2, 2, 1, 1, 3, 3]

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = colors[LR[i]-1])
        
    # 添加关节点标记
    for i in range(vals.shape[0]):
        ax.scatter(vals[i,0], vals[i,1], vals[i,2], color='red', s=50)

    RADIUS = 0.72

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if fix_z:
        left_z = max(0.0, -RADIUS+zroot)
        right_z = RADIUS+zroot
        # ax.set_zlim3d([left_z, right_z])
        ax.set_zlim3d([0, 1.5])
    else:
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])

    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D_yolov11(video_path, output_dir, model_path, debug=False, conf_thresh=0.5):
    """
    使用YOLO v11 Pose模型从视频中提取2D姿态
    
    参数:
        video_path: 视频路径
        output_dir: 输出目录
        model_path: YOLO模型路径
        debug: 是否保存转换前后的关键点对比图
        conf_thresh: 关键点置信度阈值,低于此值的关键点将使用上一帧的对应关键点
    """
    # 加载YOLO v11 Pose模型
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('\n使用YOLO v11 Pose生成2D姿态...')
    
    # 初始化关键点和分数数组
    all_keypoints = []
    all_scores = []
    
    # 处理每一帧
    for i in tqdm(range(video_length)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用YOLO v11 Pose进行检测
        results = model(frame)
        
        # 检查是否检测到人体
        if len(results) > 0 and results[0].keypoints is not None:
            # 获取第一个检测到的人的关键点
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # [17, 3] 数组，每行是[x, y, conf]
            
            # 分离坐标和置信度
            kpts_xy = keypoints[:, :2]  # [17, 2]
            scores = keypoints[:, 2]  # [17]
            
            # 应用置信度阈值过滤
            if len(all_keypoints) > 0:
                # 对于低置信度的关键点,使用上一帧的对应关键点
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
    
    cap.release()
    
    # 转换为需要的格式 [1, T, 17, 2] 和 [1, T, 17]
    keypoints_array = np.array(all_keypoints)[np.newaxis, ...]  # [1, T, 17, 2]
    scores_array = np.array(all_scores)[np.newaxis, ...]  # [1, T, 17]
    
    # 保存转换前的关键点可视化(如果debug=True)
    if debug:
        debug_dir = output_dir + 'debug/'
        os.makedirs(debug_dir, exist_ok=True)
        
        # 读取第一帧进行可视化
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # 绘制YOLOv11原始关键点
            original_vis = frame.copy()
            for j, c in enumerate([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [9, 12], [12, 13], [13, 14], [9, 15], [15, 16]]):
                pt1 = tuple(map(int, all_keypoints[0][c[0]]))
                pt2 = tuple(map(int, all_keypoints[0][c[1]]))
                cv2.line(original_vis, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(original_vis, pt1, 3, (0, 0, 255), -1)
                cv2.circle(original_vis, pt2, 3, (0, 0, 255), -1)
            
            cv2.imwrite(debug_dir + 'original_keypoints.png', original_vis)
    
    # 将YOLOv11 Pose的关键点转换为HRNet格式
    converted_keypoints = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
    converted_scores = convert_yolov11_to_hrnet_scores(scores_array.copy())
    
    # 保存转换后的关键点可视化(如果debug=True)
    if debug:
        # 读取第一帧进行可视化
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # 绘制转换后的关键点
            converted_vis = frame.copy()
            converted_vis = show2Dpose(converted_keypoints[0, 0], converted_vis)
            cv2.imwrite(debug_dir + 'converted_keypoints.png', converted_vis)
            
            # 创建并保存对比图
            comparison = np.hstack((original_vis, converted_vis))
            cv2.putText(comparison, "YOLOv11 原始关键点", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(comparison, "转换后的关键点", (original_vis.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(debug_dir + 'keypoints_comparison.png', comparison)
    
    # 更新输出数组
    keypoints_array = converted_keypoints
    scores_array = converted_scores
    
    print('使用YOLO v11 Pose生成2D姿态成功!')
    
    # 保存关键点
    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)
    output_npz = output_dir + 'input_keypoints_2d.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints_array)
    
    return keypoints_array, scores_array


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir, fix_z):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.token_num, args.layer_index = 81, 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model in 'checkpoint/pretrained/hot_mixste'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_chunks = video_length // args.frames + 1
    offset = (n_chunks * args.frames - video_length) // 2

    ret, img = cap.read()
    img_size = img.shape

    ## 3D
    print('\n生成3D姿态...')
    frame_sum = 0
    for i in tqdm(range(n_chunks)):

        ## input frames
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

        ## estimation
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

        ## h36m_cameras_extrinsic_params in common/camera.py
        # https://github.com/facebookresearch/VideoPose3D/blob/main/common/custom_dataset.py#L23
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)

        ## 2D
        for j in range(low_index, high_index):
            jj = j - frame_sum
            if i == 0 and j == 0:
                pass
            else:
                ret, img = cap.read()
                img_size = img.shape

            image = show2Dpose(input_2D_no[jj], copy.deepcopy(img))

            output_dir_2D = output_dir +'pose2D/'
            os.makedirs(output_dir_2D, exist_ok=True)
            cv2.imwrite(output_dir_2D + str(('%04d'% j)) + '_2D.png', image)

            ## 3D
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')

            post_out[jj, :, 2] -= np.min(post_out[jj, :, 2])
            show3Dpose(post_out[jj], ax, fix_z)

            output_dir_3D = output_dir +'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            plt.savefig(output_dir_3D + str(('%04d'% j)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')

        frame_sum = high_index
    
    ## save 3D keypoints
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('生成3D姿态成功!')

    ## all
    image_dir = 'results/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\n生成演示...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        edge_1 = 10
        image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("输入", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("重建", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='输入视频')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴')
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO v11 Pose模型路径')
    parser.add_argument('--debug', action='store_true', help='保存转换前后的关键点对比图')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='关键点置信度阈值,低于此值的关键点将使用上一帧的对应关键点')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'
    
    # 使用YOLO v11 Pose模型获取2D姿态
    get_pose2D_yolov11(video_path, output_dir, args.yolo_model, args.debug, args.conf_thresh)
    
    # 使用现有的3D姿态重建模型
    get_pose3D(video_path, output_dir, args.fix_z)
    
    # 生成视频
    img2video(video_path, output_dir)
    print('生成演示成功!') 