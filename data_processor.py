import numpy as np

def normalize_screen_coordinates(X, w, h):
    """
    将屏幕坐标归一化到[-1,1]范围
    
    参数:
        X: 坐标数组，形状为(..., 2)
        w: 图像宽度
        h: 图像高度
        
    返回:
        归一化后的坐标
    """
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


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
        
        # 计算双眼中点
        eyes_center = (kpts[left_eye] + kpts[right_eye]) / 2
        
        # 9: 头部(Head) - 修正的估算方法
        # 头部位置应该在"双眼连线中点与颈部中点"连线的中点位置
        new_kpts[9] = (eyes_center + neck) / 2  # 双眼中点和颈部的中点
        
        # 10: 头顶(Head top) - 修正的估算方法
        # 头顶应该在双眼连线的中点位置
        new_kpts[10] = eyes_center
        
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
        
        # 9: 头部(Head) - 修正的估算方法
        # 使用左右眼的平均置信度和颈部置信度的最小值
        eyes_conf = (s[left_eye] + s[right_eye]) / 2
        new_scores[9] = min(eyes_conf, neck_conf)
        
        # 10: 头顶(Head top) - 修正的估算方法
        # 使用左右眼的平均置信度
        new_scores[10] = eyes_conf
        
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


def camera_to_world(X, R, t):
    """
    将相机坐标系中的点转换到世界坐标系
    
    参数:
        X: 相机坐标系中的点
        R: 旋转四元数
        t: 平移向量
        
    返回:
        世界坐标系中的点
    """
    import numpy as np
    import torch
    
    def qrot(q, v):
        """
        对向量v应用四元数q表示的旋转
        
        参数:
            q: 四元数，形状为(..., 4)
            v: 向量，形状为(..., 3)
            
        返回:
            旋转后的向量
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]
        
        qvec = q[..., 1:]
        uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
        uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
        return (v + 2 * (q[..., :1] * uv + uuv))
    
    def wrap(func, *args, unsqueeze=False):
        """
        包装函数，处理numpy和torch之间的转换
        """
        args = list(args)
        for i, arg in enumerate(args):
            if type(arg) == np.ndarray:
                args[i] = torch.from_numpy(arg)
                if unsqueeze:
                    args[i] = args[i].unsqueeze(0)
        
        result = func(*args)
        
        if isinstance(result, tuple):
            result = list(result)
            for i, res in enumerate(result):
                if type(res) == torch.Tensor:
                    if unsqueeze:
                        res = res.squeeze(0)
                    result[i] = res.numpy()
            return tuple(result)
        elif type(result) == torch.Tensor:
            if unsqueeze:
                result = result.squeeze(0)
            return result.numpy()
        else:
            return result
    
    # 将相机坐标转换为世界坐标
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t 