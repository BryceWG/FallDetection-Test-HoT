from ultralytics import YOLO
import torch
import numpy as np
import cv2

# 全局变量用于存储锁定的目标信息
target_locked = False
target_bbox = None
target_id = None
frame_count = 0
MAX_INIT_FRAMES = 3  # 最多检查前三帧

def get_default_args():
    """
    返回默认参数
    """
    class Args:
        def __init__(self):
            self.confidence = 0.70
            self.nms_thresh = 0.4
            self.det_dim = 416
            self.weight_file = 'demo/lib/checkpoint/yolo11s.pt'  # 与YOLOv3保持相同的目录结构
            self.gpu = '0'
            # 新增YOLO11优化参数
            self.batch_size = 16      # 批处理大小
            self.stream = True        # 使用流模式处理
            self.vid_stride = 1       # 视频步长
            self.verbose = False      # 是否显示详细信息
            self.retina_masks = True  # 使用视网膜掩码提高精度
            self.max_det = 5         # 最大检测数量(增加以便选择面积最大的人)
    
    return Args()

def load_model(args=None, CUDA=None, inp_dim=416):
    """
    加载YOLO11模型
    """
    if args is None:
        args = get_default_args()

    if CUDA is None:
        CUDA = torch.cuda.is_available()

    # 加载YOLO11模型
    model = YOLO(args.weight_file)
    
    return model

def calculate_area(bbox):
    """
    计算边界框面积
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def reset_target():
    """
    重置目标锁定状态
    """
    global target_locked, target_bbox, target_id, frame_count
    target_locked = False
    target_bbox = None
    target_id = None
    frame_count = 0

def process_batch_frames(model, frames, args):
    """
    批量处理视频帧
    """
    global target_locked, target_bbox, target_id, frame_count
    
    results = model(
        frames,
        conf=args.confidence,          # 置信度阈值
        classes=0,                     # 只检测人类
        max_det=args.max_det,          # 增加最大检测数量以便选择面积最大的人
        verbose=args.verbose,          # 静默模式
        retina_masks=args.retina_masks,# 使用视网膜掩码
        stream=args.stream,            # 使用流模式
    )
    
    batch_bboxs = []
    batch_scores = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            # 如果目标已锁定，则使用锁定的目标
            if target_locked:
                # 尝试找到与锁定目标最接近的检测框
                best_iou = -1
                best_idx = -1
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    iou = calculate_iou(target_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                # 如果找到匹配度较高的框，使用它
                if best_iou > 0.5 and best_idx >= 0:
                    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                    bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
                    score = float(boxes.conf[best_idx].cpu().numpy())
                    target_bbox = bbox  # 更新目标框
                else:
                    # 如果没有找到匹配的框，使用上一帧的目标框
                    bbox = target_bbox
                    score = 1.0  # 默认置信度
                
                batch_bboxs.append(np.array([bbox]))
                batch_scores.append(np.array([[score]]))
            else:
                # 目标未锁定，选择面积最大的人体
                areas = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    areas.append((i, area))
                
                # 按面积降序排序
                areas.sort(key=lambda x: x[1], reverse=True)
                
                # 选择面积最大的人体
                largest_idx = areas[0][0]
                x1, y1, x2, y2 = boxes.xyxy[largest_idx].cpu().numpy()
                bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
                score = float(boxes.conf[largest_idx].cpu().numpy())
                
                # 更新帧计数
                frame_count += 1
                
                # 如果是前三帧中的第一帧检测到人体，锁定目标
                if frame_count <= MAX_INIT_FRAMES and not target_locked:
                    target_locked = True
                    target_bbox = bbox
                    print(f"已锁定目标! 帧: {frame_count}, 面积: {areas[0][1]:.2f}")
                
                batch_bboxs.append(np.array([bbox]))
                batch_scores.append(np.array([[score]]))
        else:
            # 如果当前帧未检测到人体
            if target_locked:
                # 如果目标已锁定，使用上一帧的目标框
                batch_bboxs.append(np.array([target_bbox]))
                batch_scores.append(np.array([[1.0]]))  # 默认置信度
            else:
                # 如果目标未锁定且当前帧未检测到人体
                batch_bboxs.append(None)
                batch_scores.append(None)
                
                # 更新帧计数
                frame_count += 1
                
                # 如果前三帧都未检测到人体，报错
                if frame_count >= MAX_INIT_FRAMES and not target_locked:
                    print("警告: 前三帧未检测到任何人体!")
    
    return batch_bboxs, batch_scores

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    """
    # 确保输入是numpy数组
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def yolo_human_det(img, model=None, reso=416, confidence=0.70, quiet=True):
    """
    使用YOLO11进行人体检测
    返回格式与原YOLOv3相同,保持兼容性
    Args:
        quiet: 是否静默处理（不显示检测信息）
    """
    global target_locked, target_bbox, target_id, frame_count
    
    args = get_default_args()
    args.confidence = confidence
    args.verbose = not quiet

    if model is None:
        model = load_model(args)

    if isinstance(img, str):
        img = cv2.imread(img)
    
    # 单帧处理
    if isinstance(img, (list, tuple)):
        # 批量处理
        batch_bboxs, batch_scores = process_batch_frames(model, img, args)
        return batch_bboxs, batch_scores
    else:
        # 单帧处理
        results = model(
            img,
            conf=args.confidence,          # 置信度阈值
            classes=0,                     # 只检测人类
            max_det=args.max_det,          # 增加最大检测数量以便选择面积最大的人
            verbose=args.verbose,          # 静默模式
            retina_masks=args.retina_masks,# 使用视网膜掩码
        )
        
        if len(results) == 0:
            # 如果目标已锁定，使用上一帧的目标框
            if target_locked:
                return np.array([target_bbox]), np.array([[1.0]])
            return None, None

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            # 如果目标已锁定，则使用锁定的目标
            if target_locked:
                # 尝试找到与锁定目标最接近的检测框
                best_iou = -1
                best_idx = -1
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    iou = calculate_iou(target_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                # 如果找到匹配度较高的框，使用它
                if best_iou > 0.5 and best_idx >= 0:
                    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                    bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
                    score = float(boxes.conf[best_idx].cpu().numpy())
                    target_bbox = bbox  # 更新目标框
                else:
                    # 如果没有找到匹配的框，使用上一帧的目标框
                    bbox = target_bbox
                    score = 1.0  # 默认置信度
                
                return np.array([bbox]), np.array([[score]])
            else:
                # 目标未锁定，选择面积最大的人体
                areas = []
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    areas.append((i, area))
                
                # 按面积降序排序
                areas.sort(key=lambda x: x[1], reverse=True)
                
                # 选择面积最大的人体
                largest_idx = areas[0][0]
                x1, y1, x2, y2 = boxes.xyxy[largest_idx].cpu().numpy()
                bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
                score = float(boxes.conf[largest_idx].cpu().numpy())
                
                # 更新帧计数
                frame_count += 1
                
                # 如果是前三帧中的第一帧检测到人体，锁定目标
                if frame_count <= MAX_INIT_FRAMES and not target_locked:
                    target_locked = True
                    target_bbox = bbox
                return np.array([bbox]), np.array([[score]])
        
        # 如果当前帧未检测到人体
        if target_locked:
            # 如果目标已锁定，使用上一帧的目标框
            return np.array([target_bbox]), np.array([[1.0]])
        
        # 更新帧计数
        frame_count += 1
        
        # 如果前三帧都未检测到人体，返回状态
        if frame_count >= MAX_INIT_FRAMES and not target_locked:
            return None, "No human detected"
            
        return None, None 

def yolo_human_det_v2(img, model, detector_type):
    """
    使用YOLO检测人体
    """

    # 使用全局变量跟踪状态
    global frame_count
    global target_locked
    global target_bbox
    global MAX_INIT_FRAMES
    
    # 初始化目标跟踪变量
    if 'frame_count' not in globals():
        frame_count = 0
        target_locked = False
        target_bbox = None
        MAX_INIT_FRAMES = 3
    
    # 检测人体
    results = model(img)
    
    # 获取检测结果
    if len(results.pred[0]) > 0:
        # 找到最大的人体边界框
        max_area = 0
        max_box = None
        max_conf = 0
        
        for *box, conf, cls in results.pred[0]:
            if cls == 0:  # 人体类别
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_box = box
                    max_conf = conf
        
        if max_box is not None:
            # 重置帧计数并锁定目标
            frame_count = 0
            target_locked = True
            target_bbox = max_box.cpu().numpy()
            return np.array([target_bbox]), np.array([[max_conf.cpu().numpy()]])
    
    if target_locked:
        # 如果目标已锁定，使用上一帧的目标框
        return np.array([target_bbox]), np.array([[1.0]])
    
    # 更新帧计数
    frame_count += 1
    
    # 返回None，状态信息会在主程序中处理
    if frame_count >= MAX_INIT_FRAMES and not target_locked:
        return None, "No human detected"
    return None, None