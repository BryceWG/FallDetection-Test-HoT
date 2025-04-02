# 视觉人体姿态估计与跌倒检测系统

这是一个基于视频的人体姿态估计项目，主要实现了从2D到3D的人体姿态重建过程，并使用LSTM分类器判断是否发生人体跌倒。系统支持离线视频处理和实时摄像头跌倒检测功能。

## 环境安装

```bash
# 使用uv安装PyTorch
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# 安装依赖包
uv pip install -r requirements.txt
```

## 1. 主要功能

- **2D/3D姿态处理**：从视频中提取2D人体姿态关键点并转换为3D姿态
- **姿态可视化**：生成2D和3D姿态的可视化结果
- **演示生成**：将2D和3D可视化结果并排展示，生成演示视频
- **跌倒分类**：使用LSTM模型分析姿态序列，检测跌倒行为
- **模型训练**：支持多种训练策略的LSTM跌倒检测模型训练
- **实时检测**：基于摄像头的实时跌倒检测和预警

## 2. 核心组件

### 2.1 2D姿态提取

- **检测技术**：使用YOLOv3+HRNet模型进行2D关键点检测
- **关键点**：可以输出17个人体关键点（COCO格式）
- **数据格式**：支持将2D关键点数据保存为NPZ或JSON格式
- **检测分辨率**：支持高分辨率(832×832)的人体检测

### 2.2 3D姿态预测

- **预测模型**：使用名为"hot_mixste"的模型
- **模型架构**：包含8层，512通道的深度网络
- **处理流程**：支持分块处理视频帧，并使用水平翻转进行数据增强
- **姿态映射**：从2D关键点直接映射到3D空间坐标

### 2.3 可视化模块

- **2D姿态可视化**：
  - 在原始视频帧上绘制骨架
  - 使用不同颜色区分左侧(绿色)、右侧(黄色)和中线(蓝色)
   
- **3D姿态可视化**：
  - 使用matplotlib绘制3D骨架
  - 支持固定Z轴显示
  - 提供可调整的视角参数（默认elev=15, azim=70）

- **演示生成**：
  - 将2D和3D可视化结果并排展示
  - 支持添加标题和说明文字
  - 使用多线程处理提高图像处理效率

### 2.4 跌倒检测模型

- **模型结构**：
  - 基于LSTM的序列分类模型
  - 支持单向和双向LSTM架构
  - 包含注意力机制提高序列关键帧的权重
  - 使用BatchNorm和Dropout提高泛化能力

- **训练特性**：
  - 支持Focal Loss处理类别不平衡
  - 提供标签平滑选项
  - 支持早停机制防止过拟合
  - 包含数据平衡策略（过采样、欠采样、SMOTE）
  - 支持K折交叉验证

- **数据处理**：
  - 支持2D或3D姿态数据作为输入
  - 自动计算标准化参数
  - 支持序列长度和步长自定义
  - 提供纯跌倒模式以提高训练效果

### 2.5 实时跌倒检测系统

- **多线程设计**：
  - 主线程负责摄像头采集和界面显示
  - 处理线程负责姿态提取和跌倒分析
  - 通过队列进行线程间通信

- **实时性优化**：
  - 自适应采样技术处理帧积压
  - 线性插值恢复采样后的序列完整性
  - 滑动窗口分析保持连续性

- **界面显示**：
  - 实时显示跌倒状态和概率
  - 显示系统运行状态信息
  - 支持保存检测结果视频

## 3. 使用方法

### 3.1 姿态提取和可视化

```bash
# 处理单个视频
python demo/vis.py --video sample_video.mp4 --gpu 0 --extract_2d --predict_3d --vis_2d --vis_3d --gen_demo --gen_video

# 批量处理视频
python demo/vis.py --video_dir samples --gpu 0 --all

# 自定义处理组件
python demo/vis.py --video sample_video.mp4 --extract_2d --vis_2d_first --gpu 0
```

### 3.2 LSTM模型训练

```bash
# 使用3D姿态数据训练
python train/train_lstm.py --data_dir ./demo/output/ --label_file train/frame_data.csv --pose_type 3d --lstm_bi --batch_size 16 --num_epochs 80

# 交叉验证训练
python train/train_lstm.py --data_dir ./demo/output/ --label_file train/frame_data.csv --n_splits 5 --balance_strategy smote --target_ratio 1.0 --lstm_bi

# 使用2D姿态数据训练
python train/train_lstm.py --data_dir ./demo/output/ --label_file train/frame_data.csv --pose_type 2d --batch_size 32 --num_epochs 100
```

### 3.3 实时跌倒检测

```bash
# 使用摄像头进行实时检测
python demo/fall_detect_realtime.py --camera 0 --model_dir checkpoint/fall_detection_lstm/best_model --gpu 0

# 自定义处理参数
python demo/fall_detect_realtime.py --camera 0 --model_dir checkpoint/fall_detection_lstm/best_model --buffer_size 150 --overlap_ratio 0.2 --max_pending_frames 300 --sampling_rate 0.6
```

## 4. 文件结构

```
FallDetection/
├── demo/
│   ├── vis.py                # 姿态提取和可视化
│   ├── fall_detect_realtime.py  # 实时跌倒检测
│   ├── lib/                  # 人体检测和关键点提取库
│   ├── video/                # 输入视频目录
│   └── output/               # 输出结果目录
├── train/
│   ├── train_lstm.py         # LSTM模型训练脚本
│   ├── data_balancer.py      # 数据平衡工具
│   └── frame_data.csv        # 视频帧标注数据
├── model/
│   └── mixste/               # 3D姿态预测模型
├── checkpoint/
│   └── pretrained/           # 预训练模型
├── common/
│   ├── utils.py              # 通用工具函数
│   └── camera.py             # 相机参数处理
└── README.md
```

## 5. 系统流程图

系统流程图可在 `mermaids.md` 文件中查看，包含姿态提取、模型训练和实时检测的详细流程图。