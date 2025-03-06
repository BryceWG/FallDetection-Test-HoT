import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def show2Dpose(kps, img):
    """
    在图像上绘制2D姿态
    
    参数:
        kps: 关键点坐标，形状为[17, 2]
        img: 输入图像
        
    返回:
        绘制了姿态的图像
    """
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
    """
    在3D坐标系中绘制姿态
    
    参数:
        vals: 3D关键点坐标，形状为[17, 3]
        ax: matplotlib 3D轴对象
        fix_z: 是否固定z轴范围
    """
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


def img2video(video_path, output_dir):
    """
    将图像序列转换为视频
    
    参数:
        video_path: 原始视频路径(用于获取fps)
        output_dir: 输出目录
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 获取视频名称
    video_name = video_path.split('/')[-1].split('.')[0]
    
    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    """
    在matplotlib轴上显示图像
    
    参数:
        ax: matplotlib轴对象
        img: 要显示的图像
    """
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def save_debug_images(video_path, output_dir, all_keypoints, converted_keypoints):
    """
    保存调试用的关键点对比图像
    
    参数:
        video_path: 视频路径
        output_dir: 输出目录
        all_keypoints: 原始YOLOv11关键点
        converted_keypoints: 转换后的关键点
    """
    debug_dir = output_dir + 'debug/'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 读取第一帧进行可视化
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
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
        
        # 绘制转换后的关键点
        converted_vis = frame.copy()
        converted_vis = show2Dpose(converted_keypoints[0, 0], converted_vis)
        cv2.imwrite(debug_dir + 'converted_keypoints.png', converted_vis)
        
        # 创建并保存对比图
        comparison = np.hstack((original_vis, converted_vis))
        cv2.putText(comparison, "YOLOv11 原始关键点", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "转换后的关键点", (original_vis.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(debug_dir + 'keypoints_comparison.png', comparison)
        
        # 创建专门突出显示头部关键点的图片
        head_vis = frame.copy()
        # 绘制关键点8(颈部)、9(头部中心)、10(头顶)
        neck_point = tuple(map(int, converted_keypoints[0, 0, 8]))
        head_point = tuple(map(int, converted_keypoints[0, 0, 9]))
        head_top_point = tuple(map(int, converted_keypoints[0, 0, 10]))
        
        # 绘制颈部和头部的连线
        cv2.line(head_vis, neck_point, head_point, (25, 130, 196), 2)  # 中线颜色
        cv2.line(head_vis, head_point, head_top_point, (25, 130, 196), 2)  # 中线颜色
        
        # 绘制关键点
        cv2.circle(head_vis, neck_point, 5, (0, 0, 255), -1)
        cv2.circle(head_vis, head_point, 5, (0, 0, 255), -1)
        cv2.circle(head_vis, head_top_point, 5, (0, 0, 255), -1)
        
        # 添加标签
        cv2.putText(head_vis, "颈部(8)", (neck_point[0]+10, neck_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(head_vis, "头部中心(9)", (head_point[0]+10, head_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(head_vis, "头顶(10)", (head_top_point[0]+10, head_top_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imwrite(debug_dir + 'head_keypoints.png', head_vis)
        
        # 保存更多帧进行调试
        for i in range(1, min(5, len(all_keypoints))):  # 保存第1-4帧(第0帧已经保存)
            ret, frame = cap.read()
            if not ret:
                break
                
            # 绘制当前帧的原始关键点
            frame_vis_orig = frame.copy()
            for j, c in enumerate([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [9, 12], [12, 13], [13, 14], [9, 15], [15, 16]]):
                pt1 = tuple(map(int, all_keypoints[i][c[0]]))
                pt2 = tuple(map(int, all_keypoints[i][c[1]]))
                cv2.line(frame_vis_orig, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(frame_vis_orig, pt1, 3, (0, 0, 255), -1)
                cv2.circle(frame_vis_orig, pt2, 3, (0, 0, 255), -1)
            
            cv2.imwrite(debug_dir + f'frame_{i}_original.png', frame_vis_orig)
            
            # 绘制当前帧的转换后关键点
            frame_vis_conv = frame.copy()
            frame_vis_conv = show2Dpose(converted_keypoints[0, i], frame_vis_conv)
            cv2.imwrite(debug_dir + f'frame_{i}_converted.png', frame_vis_conv)
    
    cap.release()


def create_visualization(output_dir):
    """
    创建最终的可视化结果
    
    参数:
        output_dir: 输出目录
    """
    # 获取2D和3D姿态图像路径
    output_dir_2D = output_dir + 'pose2D/'
    output_dir_3D = output_dir + 'pose3D/'
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    # 创建输出目录
    output_dir_pose = output_dir + 'pose/'
    os.makedirs(output_dir_pose, exist_ok=True)

    # 生成可视化
    for i in range(len(image_2d_dir)):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # 裁剪图像
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
        edge_1 = 10
        image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        # 创建图像
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("输入", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("重建", fontsize = font_size)

        # 保存图像
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close() 