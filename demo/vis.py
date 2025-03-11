import os 
import sys
import cv2
import glob
import copy
import json
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
import time
import concurrent.futures

def show2Dpose(kps, img):
    """
    在图像上绘制2D人体姿态骨架
    """
    # 定义颜色: 绿色(左侧), 黄色(右侧), 蓝色(中线)
    colors = [(138, 201, 38),    # 绿色 - 左侧
              (255, 202, 58),    # 黄色 - 右侧
              (25, 130, 196)]    # 蓝色 - 中线

    # 定义连接关系
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    # 定义左右侧: 0=左侧(绿色), 1=右侧(黄色), 2=中线(蓝色)
    # 根据人体解剖学正确分配颜色:
    # 0-1, 1-2, 2-3: 右腿 - 右侧色
    # 0-4, 4-5, 5-6: 左腿 - 左侧色
    # 0-7, 7-8, 8-9, 9-10: 脊柱和头部 - 中线色
    # 8-11, 11-12, 12-13: 左臂 - 左侧色
    # 8-14, 14-15, 15-16: 右臂 - 右侧色
    LR = [1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1]

    thickness = 3

    # 首先绘制连接线
    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]], radius=3)

    return img


def show3Dpose(vals, ax, fix_z):
    """
    在3D坐标系中绘制人体姿态骨架
    """
    ax.view_init(elev=15., azim=70)

    # 定义颜色: 绿色(左侧), 黄色(右侧), 蓝色(中线)
    colors = [(138/255, 201/255, 38/255),  # 绿色 - 左侧
              (255/255, 202/255, 58/255),  # 黄色 - 右侧
              (25/255, 130/255, 196/255)]  # 蓝色 - 中线

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    # 定义左右侧: 0=左侧(绿色), 1=右侧(黄色), 2=中线(蓝色)
    # 根据人体解剖学正确分配颜色:
    # 0-1, 0-4: 髋部连接
    # 1-2, 2-3: 右腿
    # 4-5, 5-6: 左腿
    # 0-7, 7-8, 8-9, 9-10: 脊柱和头部
    # 8-11, 11-12, 12-13: 左臂
    # 8-14, 14-15, 15-16: 右臂
    LR = [2, 0, 2, 0, 2, 0, 1, 1, 2, 0, 2, 2, 0, 0, 1, 1]

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = colors[LR[i]])

    # 添加关节点标记,使用红色增强可见度
    ax.scatter(vals[:,0], vals[:,1], vals[:,2], c='red', marker='o', s=50)

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


def get_pose2D(video_path, output_dir, save_json=False, detector='yolov3'):
    """
    从视频中提取2D人体姿态关键点
    分两个阶段：
    1. YOLO人体检测
    2. HRNet姿态估计
    """
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\n开始2D姿态生成过程...')
    with torch.no_grad():
        # 根据选择的检测器加载相应的模型
        if detector == 'yolo11':
            from lib.yolo11.human_detector import load_model as yolo_model
            from lib.yolo11.human_detector import yolo_human_det as yolo_det
            print('使用YOLO11检测器')
        else:  # 默认使用YOLOv3
            from lib.yolov3.human_detector import load_model as yolo_model
            from lib.yolov3.human_detector import yolo_human_det as yolo_det
            print('使用YOLOv3检测器')
            
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True, detector=detector)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('2D姿态生成完成!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'input_keypoints_2d.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    
    if save_json:
        json_data = []
        print('\n正在生成JSON格式数据...')
        for frame_idx, frame_kpts in enumerate(keypoints[0]):
            flattened_keypoints = []
            for kpt_idx in range(len(frame_kpts)):
                x, y = frame_kpts[kpt_idx][0], frame_kpts[kpt_idx][1]
                c = scores[0][frame_idx][kpt_idx] if scores is not None else 1.0
                flattened_keypoints.extend([float(x), float(y), float(c)])
            
            json_data.append({
                "idx": 0,
                "keypoints": flattened_keypoints
            })
        
        with open(output_dir + 'keypoints_2d.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print('JSON数据保存完成:', output_dir + 'keypoints_2d.json')
    
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
    使用OpenCV代替matplotlib以提高速度
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
    output_dir_pose = output_dir + 'pose/'
    os.makedirs(output_dir_pose, exist_ok=True)
    
    for i in tqdm(range(min(len(image_2d_dir), len(image_3d_dir)))):
        # 使用OpenCV读取图像
        image_2d = cv2.imread(image_2d_dir[i])
        image_3d = cv2.imread(image_3d_dir[i])
        
        # 转换为RGB颜色空间（如果需要）
        image_2d = cv2.cvtColor(image_2d, cv2.COLOR_BGR2RGB)
        image_3d = cv2.cvtColor(image_3d, cv2.COLOR_BGR2RGB)

        # 裁剪
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
        edge_1 = 10
        image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]
        
        # 调整图像大小以确保两张图片高度相同
        h1, w1 = image_2d.shape[:2]
        h2, w2 = image_3d.shape[:2]
        target_height = max(h1, h2)
        
        # 调整图像大小，保持纵横比
        new_w1 = int(w1 * (target_height / h1))
        new_w2 = int(w2 * (target_height / h2))
        
        image_2d = cv2.resize(image_2d, (new_w1, target_height))
        image_3d = cv2.resize(image_3d, (new_w2, target_height))
        
        # 创建标题
        title_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        title_color = (0, 0, 0)  # 黑色
        
        # 为图像添加标题空间
        img_2d_with_title = np.ones((target_height + title_height, new_w1, 3), dtype=np.uint8) * 255
        img_3d_with_title = np.ones((target_height + title_height, new_w2, 3), dtype=np.uint8) * 255
        
        # 添加标题
        cv2.putText(img_2d_with_title, "Input", (10, title_height - 10), font, font_scale, title_color, font_thickness)
        cv2.putText(img_3d_with_title, "Reconstruction", (10, title_height - 10), font, font_scale, title_color, font_thickness)
        
        # 将图像放入带标题的画布中
        img_2d_with_title[title_height:, :, :] = image_2d
        img_3d_with_title[title_height:, :, :] = image_3d
        
        # 水平拼接两张图像
        combined_img = np.hstack((img_2d_with_title, img_3d_with_title))
        
        # 转换回BGR颜色空间用于保存
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        
        # 保存结果
        output_path = output_dir_pose + str(('%04d'% i)) + '_pose.png'
        cv2.imwrite(output_path, combined_img)
    
    print('演示生成成功!')
    return output_dir + 'pose/'


def img2video(video_path, output_dir):
    """
    将图像序列转换为视频
    使用多线程加速处理
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_name = video_path.split('/')[-1].split('.')[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        print('未找到图像序列,请先生成演示')
        return
        
    # 读取第一张图片获取尺寸
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    print('\n正在生成视频...')
    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    # 定义一个函数用于读取图像
    def read_image(name):
        return cv2.imread(name)
    
    # 使用线程池并行读取图像
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有读取任务并获取future对象
        future_to_name = {executor.submit(read_image, name): name for name in names}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(names), desc="读取图像"):
            name = future_to_name[future]
            try:
                # 获取结果并保存到列表中
                img = future.result()
                # 保存图像和对应的索引，以便后续按顺序写入
                index = names.index(name)
                frames.append((index, img))
            except Exception as exc:
                print(f'{name} 读取失败: {exc}')
    
    # 按原始顺序排序帧
    frames.sort(key=lambda x: x[0])
    
    # 写入视频
    for _, img in tqdm(frames, desc="写入视频"):
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
    parser.add_argument('--video_dir', type=str, default=None, help='输入视频目录(用于批量处理)')
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
    parser.add_argument('--2d_json', action='store_true', help='将2D姿态数据以JSON格式输出')
    parser.add_argument('--detector', type=str, default='yolov3', choices=['yolov3', 'yolo11'], help='选择人体检测器')
    
    # 解析已知参数,忽略未知参数(这些参数可能是HRNet的)
    args, unknown = parser.parse_known_args()
    return args


def process_single_video(video_path, args, start_time=None):
    """
    处理单个视频文件
    """
    print(f'\n处理视频: {video_path}')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = './demo/output/' + video_name + '/'
    
    # 提取2D姿态
    keypoints = None
    if args.extract_2d:
        keypoints = get_pose2D(video_path, output_dir, save_json=getattr(args, '2d_json', False), detector=args.detector)
    
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
    
    # 显示该视频的处理时间
    if start_time:
        current_time = time.time()
        elapsed_time = current_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f'视频 {video_name} 处理完成! 耗时: {int(minutes)}分{seconds:.2f}秒')


if __name__ == "__main__":
    # 处理命令行参数
    args = process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 如果没有选择任何功能,默认执行所有步骤
    if not (args.extract_2d or args.predict_3d or args.vis_2d or args.vis_3d or args.gen_demo or args.gen_video):
        args.all = True
    
    # 执行所有步骤
    if args.all:
        args.extract_2d = args.predict_3d = args.vis_2d = args.vis_3d = args.gen_demo = args.gen_video = True
    
    # 开始计时
    start_time = time.time()

    # 处理单个视频或批量处理视频
    if args.video_dir:
        # 获取视频目录下所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join('./demo/video/', args.video_dir, f'*{ext}')))
        
        if not video_files:
            print(f'错误: 在目录 {args.video_dir} 中未找到视频文件')
            sys.exit(1)
        
        # 显示找到的视频文件
        print(f'找到 {len(video_files)} 个视频文件:')
        for video_file in video_files:
            print(f'- {os.path.basename(video_file)}')
        
        # 批量处理视频
        for i, video_path in enumerate(video_files, 1):
            print(f'\n处理第 {i}/{len(video_files)} 个视频')
            process_single_video(video_path, args, start_time)
    else:
        # 处理单个视频
        video_path = './demo/video/' + args.video
        process_single_video(video_path, args, start_time)
    
    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f'\n总耗时: {int(minutes)}分{seconds:.2f}秒')
    
    if args.all:
        print('所有步骤执行完成!')