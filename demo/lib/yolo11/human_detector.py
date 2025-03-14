from ultralytics import YOLO
import torch
import numpy as np
import cv2

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
            self.max_det = 1          # 最大检测数量(因为我们只关注一个人)
    
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

def process_batch_frames(model, frames, args):
    """
    批量处理视频帧
    """
    results = model(
        frames,
        conf=args.confidence,          # 置信度阈值
        classes=0,                     # 只检测人类
        max_det=args.max_det,         # 限制每帧最大检测数
        verbose=args.verbose,          # 静默模式
        retina_masks=args.retina_masks,# 使用视网膜掩码
        stream=args.stream,            # 使用流模式
    )
    
    batch_bboxs = []
    batch_scores = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # 获取最高置信度的检测结果
            boxes = result.boxes
            conf_idx = boxes.conf.argmax() if len(boxes) > 1 else 0
            
            # 获取边界框坐标
            x1, y1, x2, y2 = boxes.xyxy[conf_idx].cpu().numpy()
            bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
            score = float(boxes.conf[conf_idx].cpu().numpy())
            
            batch_bboxs.append(np.array([bbox]))
            batch_scores.append(np.array([[score]]))
        else:
            batch_bboxs.append(None)
            batch_scores.append(None)
    
    return batch_bboxs, batch_scores

def yolo_human_det(img, model=None, reso=416, confidence=0.70, quiet=True):
    """
    使用YOLO11进行人体检测
    返回格式与原YOLOv3相同,保持兼容性
    Args:
        quiet: 是否静默处理（不显示检测信息）
    """
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
            max_det=args.max_det,         # 限制最大检测数
            verbose=args.verbose,          # 静默模式
            retina_masks=args.retina_masks,# 使用视网膜掩码
        )
        
        if len(results) == 0:
            return None, None

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            # 获取最高置信度的检测结果
            boxes = result.boxes
            conf_idx = boxes.conf.argmax() if len(boxes) > 1 else 0
            
            # 获取边界框坐标
            x1, y1, x2, y2 = boxes.xyxy[conf_idx].cpu().numpy()
            bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
            score = float(boxes.conf[conf_idx].cpu().numpy())
            
            return np.array([bbox]), np.array([[score]])
        
        return None, None 