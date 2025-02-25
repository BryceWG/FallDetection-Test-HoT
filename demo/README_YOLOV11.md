# 使用YOLO v11 Pose进行姿态估计

本文档介绍如何使用YOLO v11 Pose模型替代原有的YOLO v3和HRNet组合，以简化姿态估计流程并提高性能。

## 优势

1. **简化流程**：YOLO v11 Pose可以一次性完成人体检测和姿态估计，无需分开处理
2. **更小的模型**：YOLO v11 Pose模型参数量更小，推理速度更快
3. **更高的准确性**：YOLO v11 Pose采用了最新的姿态估计技术，准确性更高
4. **更简单的使用方式**：使用ultralytics库提供的简洁API

## 安装依赖

```bash
pip install ultralytics
```

## 准备模型

1. 下载YOLO v11 Pose模型（如果尚未下载）
   - 可以使用`yolo11n-pose.pt`（小模型）或`yolo11s-pose.pt`（中等大小模型）
   - 将模型文件放在项目根目录下

## 使用方法

使用新的`vis_yolov11.py`脚本替代原有的`vis.py`：

```bash
python demo/vis_yolov11.py --video sample_video.mp4 --yolo_model yolo11n-pose.pt
```

参数说明：
- `--video`：输入视频文件名（位于`./demo/video/`目录下）
- `--yolo_model`：YOLO v11 Pose模型文件路径
- `--gpu`：指定GPU ID（默认为0）
- `--fix_z`：固定z轴（可选）

## 工作流程

1. 使用YOLO v11 Pose模型从视频中提取人体关键点
2. 使用现有的3D姿态重建模型将2D关键点转换为3D姿态
3. 生成可视化结果和演示视频

## 注意事项

1. YOLO v11 Pose和原有模型的关键点定义可能略有不同，但代码已经处理了这些差异
2. 如果处理视频时遇到内存问题，可以考虑使用更小的YOLO模型（如`yolo11n-pose.pt`）
3. 确保您的CUDA和PyTorch版本兼容

## 对比原有方法

原有方法：
1. 使用YOLO v3检测人体 → 2. 使用HRNet提取关键点 → 3. 使用3D姿态重建模型

新方法：
1. 使用YOLO v11 Pose同时检测人体和提取关键点 → 2. 使用3D姿态重建模型

这种简化不仅提高了处理速度，还减少了模型加载和推理的复杂性。 