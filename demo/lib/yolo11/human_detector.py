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
            self.weight_file = 'demo/lib/checkpoint/yolo11n.pt'  # 与YOLOv3保持相同的目录结构
            self.gpu = '0'
    
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

def yolo_human_det(img, model=None, reso=416, confidence=0.70):
    """
    使用YOLO11进行人体检测
    返回格式与原YOLOv3相同,保持兼容性
    """
    args = get_default_args()
    args.confidence = confidence

    if model is None:
        model = load_model(args)

    if isinstance(img, str):
        img = cv2.imread(img)

    # 使用YOLO11进行预测
    results = model(img, conf=args.confidence, classes=0)  # class 0 是'person'类别
    
    if len(results) == 0:
        return None, None

    # 提取边界框和置信度
    bboxs = []
    scores = []
    
    # 处理第一张图片的结果
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # 转换为[x1, y1, x2, y2]格式并四舍五入到2位小数
            bbox = [round(float(x), 2) for x in [x1, y1, x2, y2]]
            bboxs.append(bbox)
            # 获取置信度分数
            score = float(box.conf[0].cpu().numpy())
            scores.append(score)
    
    if not bboxs:  # 如果没有检测到人
        return None, None
    
    # 转换为numpy数组,保持与原接口一致的格式
    bboxs = np.array(bboxs)
    scores = np.expand_dims(np.array(scores), 1)
    
    return bboxs, scores 