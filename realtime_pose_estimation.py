import os
import cv2
import time
import argparse
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

# 导入自定义模块
from data_processor import convert_yolov11_to_hrnet_keypoints
from render_3d_poses import render_single_frame # 用于生成3D渲染图
import matplotlib.pyplot as plt

# 导入mixste模型模块
from model.mixste.hot_mixste import Model

def init_yolo(yolo_model_path):
    print("加载 YOLO v11 Pose 模型...")
    model = YOLO(yolo_model_path)
    return model

def init_mixste():
    print("初始化 mixste 3D 姿态预测模型...")
    # 构造模型参数（依据vis_yolov11中设定参数）
    class DummyArgs:
        pass
    
    args = DummyArgs()
    args.layers = 8
    args.channel = 512
    args.d_hid = 1024
    args.frames = 243 # 模型输入帧数
    args.token_num = 81
    args.layer_index = 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    args.n_joints = 17
    args.out_joints = 17

    model = Model(args).cuda()

    # 加载预训练模型
    import glob
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
    if not model_paths:
        raise ValueError("未找到 mixste 预训练模型，请检查路径：{}".format(args.previous_dir))

    model_path = model_paths[0]
    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model, args

def predict_2d_pose(frame, yolo_model, conf_thresh=0.5):
    """
    调用YOLO模型进行2D姿态提取，返回HRNet格式的2D关键点
    """
    results = yolo_model(frame)
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy() # shape: [17,3]
        kpts_xy = keypoints[:, :2]
        scores = keypoints[:, 2]
        # 此处可根据置信度处理低置信度关键点，这里简化处理
    else:
        # 未检测到则返回17个零点
        kpts_xy = np.zeros((17, 2))
        print("未检测到人体，返回零关键点")

    # 确保关键点数组形状正确
    if kpts_xy.shape != (17, 2):
        print(f"警告：关键点形状不正确 {kpts_xy.shape}，重置为零关键点")
        kpts_xy = np.zeros((17, 2))

    # 将单帧2D姿态转为HRNet格式（扩展维度为[T,17,2]）
    kpts_hrnet = convert_yolov11_to_hrnet_keypoints(np.expand_dims(kpts_xy, axis=0))[0]
    return kpts_hrnet

def predict_3d_pose(buffer_2d, mixste_model, args):
    """
    输入2D关键点序列缓冲区 (T, 17,2) 转换为模型输入格式，并调用mixste模型预测3D姿态.
    注意：输入buffer_2d须为固定长度 args.frames.
    """
    # 归一化输入（这里需要摄像头尺寸，需要自行指定，假设摄像头分辨率为W,H）
    # 此处简化处理直接使用2D关键点，实际应用可调用normalize_screen_coordinates
    input_2d = buffer_2d.copy() # shape: (T,17,2)

    # 构造镜像数据以增强输入，如vis_yolov11中所做
    joints_left = [4,5,6,11,12,13]
    joints_right = [1,2,3,14,15,16]
    input_2d_aug = input_2d.copy()
    input_2d_aug[:, :, 0] = -1
    input_2d_aug[:, joints_left + joints_right] = input_2d_aug[:, joints_right + joints_left]

    # 合并两个来源，shape变为(2, T,17,2)
    input_2d = np.stack([input_2d, input_2d_aug], axis=0)

    # 再添加batch维度: (1,2, T,17,2) -> 模型期望输入 (B,2, T,17,2)
    input_2d = np.expand_dims(input_2d, axis=0)
    input_2d = torch.from_numpy(input_2d.astype('float32')).cuda()

    with torch.no_grad():
        output_3d_non_flip = mixste_model(input_2d[:, 0])
        output_3d_flip = mixste_model(input_2d[:, 1])
        output_3d_flip[:, :, :, 0] = -1
        output_3d_flip[:, :, joints_left + joints_right, :] = output_3d_flip[:, :, joints_right + joints_left, :]
        output_3d = (output_3d_non_flip + output_3d_flip) / 2

    # 去除多余batch维度，返回形状为 (T,17,3) 的3D预测
    output_3d = output_3d[0].cpu().detach().numpy()
    return output_3d

def update_3d_display(pose_3d, disp_window='3D Pose'):
    """
    使用render_single_frame方法渲染最新一帧3D姿态，并在OpenCV窗口中显示
    """
    temp_img_path = "./temp_3d.png"
    # 注意：render_single_frame内部也要进行坐标转换，这里保持一致性
    render_single_frame(pose_3d, -1, temp_img_path, fix_z=True, view_angles=(15,70))
    img_3d = cv2.imread(temp_img_path)
    if img_3d is not None:
        cv2.imshow(disp_window, img_3d)

def main():
    parser = argparse.ArgumentParser(description="实时姿态预测流水线")
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt', help='YOLO v11 Pose模型路径')
    parser.add_argument('--display', action='store_true', help='启用UI界面实时显示2D和3D画面')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='2D关键点置信度阈值')
    parser.add_argument('--output', type=str, default='real_time_3d_pose.npz', help='保存3D预测结果的npz文件')
    args = parser.parse_args()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化模型
    yolo_model = init_yolo(args.yolo_model)
    mixste_model, mixste_args = init_mixste()

    # 用于保存2D和3D预测数据
    saved_3d_poses = []

    # 建立一个固定长度的滑动窗口，存储最近帧的2D关键点，用于3D预测
    buffer_length = mixste_args.frames # 例如243帧
    buffer_2d = deque(maxlen=buffer_length)

    # 创建基础摄像头窗口（用于确认输入）
    cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
    
    # 如果启用显示，额外创建带标注的摄像头窗口和3D姿态窗口
    if args.display:
        cv2.namedWindow("2D Pose", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3D Pose", cv2.WINDOW_NORMAL)

    print("开始实时预测，按 'q' 键退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 始终显示基础摄像头画面（不带标注）
        cv2.imshow("Camera Input", frame)

        # 2D姿态预测
        pose2d = predict_2d_pose(frame, yolo_model, args.conf_thresh)
        # 将2D姿态存入buffer
        buffer_2d.append(pose2d)

        # 若显示模式开启，显示带标注的2D姿态和3D姿态
        if args.display:
            # 创建带标注的2D显示帧
            display_frame = frame.copy()
            # 绘制关键点和骨架
            for (x, y) in pose2d:
                cv2.circle(display_frame, (int(x), int(y)), 3, (0,0,255), -1)
            cv2.imshow("2D Pose", display_frame)

        # 每当2D预测buffer填满，则进行3D姿态预测
        if len(buffer_2d) == buffer_length:
            # 将buffer转换为numpy数组形状 (T,17,2)
            seq_2d = np.array(buffer_2d)
            # 进行3D姿态预测
            pose3d_seq = predict_3d_pose(seq_2d, mixste_model, mixste_args)
            # 保存最新一帧的3D预测结果
            latest_pose3d = pose3d_seq[-1]
            saved_3d_poses.append(latest_pose3d)

            # 如果显示开启，更新3D姿态显示
            if args.display:
                temp_seq = np.expand_dims(pose3d_seq[-1], axis=0)
                render_single_frame(temp_seq, 0, "./temp_3d.png", fix_z=True, view_angles=(15,70))
                img_3d = cv2.imread("./temp_3d.png")
                if img_3d is not None:
                    cv2.imshow("3D Pose", img_3d)

        # 检查退出条件（按 'q' 键退出）- 使用基础摄像头窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 保存3D预测结果到npz文件
    saved_3d_poses = np.array(saved_3d_poses)
    np.savez_compressed(args.output, reconstruction=saved_3d_poses)
    print("3D姿态预测结果已保存到：", args.output)

if __name__ == "__main__":
    main()