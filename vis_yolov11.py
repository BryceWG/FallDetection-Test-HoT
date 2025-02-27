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
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.getcwd())
from common.utils import *
from model.mixste.hot_mixste import Model

# 导入自定义模块
from data_processor import normalize_screen_coordinates, convert_yolov11_to_hrnet_keypoints, convert_yolov11_to_hrnet_scores, camera_to_world
from utils import show2Dpose, show3Dpose, img2video, showimage, save_debug_images, create_visualization


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
        # 将YOLOv11 Pose的关键点转换为HRNet格式
        converted_keypoints = convert_yolov11_to_hrnet_keypoints(keypoints_array.copy())
        # 保存调试图像
        save_debug_images(video_path, output_dir, all_keypoints, converted_keypoints)
    
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


def get_pose3D(video_path, output_dir, fix_z, fast_mode=False):
    """
    从2D姿态生成3D姿态
    
    参数:
        video_path: 视频路径
        output_dir: 输出目录
        fix_z: 是否固定z轴范围
        fast_mode: 快速模式，只进行3D姿态预测，不生成可视化结果
    """
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

    ret, img = cap.read()
    img_size = img.shape

    ## 生成3D姿态
    print('\n生成3D姿态...')
    frame_sum = 0
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

        # 如果不是快速模式，则进行可视化
        if not fast_mode:
            ## 坐标系转换
            # h36m_cameras_extrinsic_params in common/camera.py
            # https://github.com/facebookresearch/VideoPose3D/blob/main/common/custom_dataset.py#L23
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)

            ## 生成2D和3D可视化
            for j in range(low_index, high_index):
                jj = j - frame_sum
                if i == 0 and j == 0:
                    pass
                else:
                    ret, img = cap.read()
                    img_size = img.shape

                # 2D姿态可视化
                image = show2Dpose(input_2D_no[jj], copy.deepcopy(img))

                output_dir_2D = output_dir +'pose2D/'
                os.makedirs(output_dir_2D, exist_ok=True)
                cv2.imwrite(output_dir_2D + str(('%04d'% j)) + '_2D.png', image)

                # 3D姿态可视化
                fig = plt.figure(figsize=(9.6, 5.4))
                gs = gridspec.GridSpec(1, 1)
                gs.update(wspace=-0.00, hspace=0.05) 
                ax = plt.subplot(gs[0], projection='3d')

                post_out[jj, :, 2] -= np.min(post_out[jj, :, 2])
                show3Dpose(post_out[jj], ax, fix_z)

                output_dir_3D = output_dir +'pose3D/'
                os.makedirs(output_dir_3D, exist_ok=True)
                plt.savefig(output_dir_3D + str(('%04d'% j)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
                plt.close()

        frame_sum = high_index
    
    ## 保存3D关键点
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('生成3D姿态成功!')
    
    # 如果不是快速模式，创建最终可视化
    if not fast_mode:
        create_visualization(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='输入视频')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--fix_z', action='store_true', help='固定z轴')
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO v11 Pose模型路径')
    parser.add_argument('--debug', action='store_true', help='保存转换前后的关键点对比图')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='关键点置信度阈值,低于此值的关键点将使用上一帧的对应关键点')
    parser.add_argument('--fast_mode', action='store_true', help='快速模式，只进行3D姿态预测，不生成可视化结果')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'
    
    # 使用YOLO v11 Pose模型获取2D姿态
    get_pose2D_yolov11(video_path, output_dir, args.yolo_model, args.debug, args.conf_thresh)
    
    # 使用现有的3D姿态重建模型
    get_pose3D(video_path, output_dir, args.fix_z, args.fast_mode)
    
    # 如果不是快速模式，生成视频
    if not args.fast_mode:
        img2video(video_path, output_dir)
        print('生成演示成功!')
    else:
        print('快速模式完成：3D姿态数据已保存到', output_dir + 'output_3D/output_keypoints_3d.npz')


if __name__ == "__main__":
    main() 