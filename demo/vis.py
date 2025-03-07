import os 
import sys
import cv2
import glob
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from IPython import embed

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


def show2Dpose(kps, img):
    """
    在图像上绘制2D人体姿态骨架
    """
    colors = [(138, 201, 38),
              (25, 130, 196),
              (255, 202, 58)] 

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)

    return img


def show3Dpose(vals, ax, fix_z):
    """
    在3D坐标系中绘制人体姿态骨架
    """
    ax.view_init(elev=15., azim=70)

    colors = [(138/255, 201/255, 38/255),
            (255/255, 202/255, 58/255),
            (25/255, 130/255, 196/255)]

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = colors[LR[i]-1])

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


def get_pose2D(video_path, output_dir):
    """
    从视频中提取2D人体姿态关键点
    """
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\n正在生成2D姿态...')
    with torch.no_grad():
        # 视频的第一帧应该检测到一个人
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('2D姿态生成成功!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'input_keypoints_2d.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    
    return keypoints


def visualize_pose2D(video_path, output_dir, keypoints=None):
    """
    可视化2D姿态并保存结果
    """
    if keypoints is None:
        # 如果没有提供关键点,则从保存的文件加载
        keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']
    
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('\n正在可视化2D姿态...')
    output_dir_2D = output_dir + 'pose2D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if not ret:
            break
            
        image = show2Dpose(keypoints[0][i], copy.deepcopy(img))
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)
    
    print('2D姿态可视化完成!')
    return output_dir_2D


def get_pose3D(video_path, output_dir, fix_z):
    """
    从2D姿态预测3D姿态
    """
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.token_num, args.layer_index = 81, 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    args.n_joints, args.out_joints = 17, 17

    # 加载模型
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # 预训练模型放在 'checkpoint/pretrained/hot_mixste'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model.eval()

    # 加载2D关键点
    keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_chunks = video_length // args.frames + 1
    offset = (n_chunks * args.frames - video_length) // 2

    ret, img = cap.read()
    img_size = img.shape

    # 3D姿态预测
    print('\n正在生成3D姿态...')
    frame_sum = 0
    output_3d_all = None
    
    for i in tqdm(range(n_chunks)):
        # 输入帧
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

        # 预测
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

        # 坐标系转换
        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        
        frame_sum = high_index
    
    # 保存3D关键点
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('3D姿态生成成功!')
    return output_3d_all


def visualize_pose3D(video_path, output_dir, fix_z, output_3d_all=None):
    """
    可视化3D姿态并保存结果
    """
    if output_3d_all is None:
        # 如果没有提供3D关键点,则从保存的文件加载
        output_3d_all = np.load(output_dir + 'output_3D/output_keypoints_3d.npz', allow_pickle=True)['reconstruction']
    
    # 坐标系转换
    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(output_3d_all, R=rot, t=0)
    
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('\n正在可视化3D姿态...')
    output_dir_3D = output_dir + 'pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    
    for i in tqdm(range(min(video_length, len(post_out)))):
        # 3D可视化
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')

        post_out[i, :, 2] -= np.min(post_out[i, :, 2])
        show3Dpose(post_out[i], ax, fix_z)

        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close()
    
    print('3D姿态可视化完成!')
    return output_dir_3D


def generate_demo(output_dir):
    """
    生成最终的演示视频,将2D和3D姿态并排展示
    """
    # 获取已生成的2D和3D姿态图像
    output_dir_2D = output_dir + 'pose2D/'
    output_dir_3D = output_dir + 'pose3D/'
    
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))
    
    if not image_2d_dir or not image_3d_dir:
        print('未找到2D或3D姿态图像,请先生成它们')
        return
    
    print('\n正在生成演示...')
    for i in tqdm(range(min(len(image_2d_dir), len(image_3d_dir)))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # 裁剪
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
        edge_1 = 10
        image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        # 显示
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        # 保存
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close()
    
    print('演示生成成功!')
    return output_dir + 'pose/'


def img2video(video_path, output_dir):
    """
    将图像序列转换为视频
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    video_name = video_path.split('/')[-1].split('.')[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        print('未找到图像序列,请先生成演示')
        return
        
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    print('\n正在生成视频...')
    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in tqdm(names):
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()
    print(f'视频生成成功: {output_dir + video_name}.mp4')


def showimage(ax, img):
    """
    在matplotlib轴上显示图像
    """
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def process_args():
    """
    处理命令行参数,与HRNet的参数分开处理
    """
    parser = argparse.ArgumentParser(description='3D姿态估计演示')
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='输入视频')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--fix_z', action='store_true', help='固定Z轴')
    
    # 添加功能选择参数
    parser.add_argument('--extract_2d', action='store_true', help='提取2D姿态')
    parser.add_argument('--predict_3d', action='store_true', help='预测3D姿态')
    parser.add_argument('--vis_2d', action='store_true', help='可视化2D姿态')
    parser.add_argument('--vis_3d', action='store_true', help='可视化3D姿态')
    parser.add_argument('--gen_demo', action='store_true', help='生成演示(2D和3D并排)')
    parser.add_argument('--gen_video', action='store_true', help='生成视频')
    parser.add_argument('--all', action='store_true', help='执行所有步骤')
    
    # 解析已知参数,忽略未知参数(这些参数可能是HRNet的)
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # 处理命令行参数
    args = process_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'
    
    # 如果没有选择任何功能,默认执行所有步骤
    if not (args.extract_2d or args.predict_3d or args.vis_2d or args.vis_3d or args.gen_demo or args.gen_video):
        args.all = True
    
    # 执行所有步骤
    if args.all:
        args.extract_2d = args.predict_3d = args.vis_2d = args.vis_3d = args.gen_demo = args.gen_video = True
    
    # 提取2D姿态
    keypoints = None
    if args.extract_2d:
        keypoints = get_pose2D(video_path, output_dir)
    
    # 可视化2D姿态
    if args.vis_2d:
        if not os.path.exists(output_dir + 'input_2D/input_keypoints_2d.npz') and keypoints is None:
            print('未找到2D关键点,请先提取2D姿态')
        else:
            visualize_pose2D(video_path, output_dir, keypoints)
    
    # 预测3D姿态
    output_3d_all = None
    if args.predict_3d:
        if not os.path.exists(output_dir + 'input_2D/input_keypoints_2d.npz') and keypoints is None:
            print('未找到2D关键点,请先提取2D姿态')
        else:
            output_3d_all = get_pose3D(video_path, output_dir, args.fix_z)
    
    # 可视化3D姿态
    if args.vis_3d:
        if not os.path.exists(output_dir + 'output_3D/output_keypoints_3d.npz') and output_3d_all is None:
            print('未找到3D关键点,请先预测3D姿态')
        else:
            visualize_pose3D(video_path, output_dir, args.fix_z, output_3d_all)
    
    # 生成演示
    if args.gen_demo:
        if not os.path.exists(output_dir + 'pose2D/') or not os.path.exists(output_dir + 'pose3D/'):
            print('未找到2D或3D姿态图像,请先生成它们')
        else:
            generate_demo(output_dir)
    
    # 生成视频
    if args.gen_video:
        if not os.path.exists(output_dir + 'pose/'):
            print('未找到演示图像,请先生成演示')
        else:
            img2video(video_path, output_dir)
    
    if args.all:
        print('所有步骤执行完成!')


