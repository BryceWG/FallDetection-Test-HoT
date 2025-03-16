# 跌倒检测LSTM模型训练使用说明

## 项目简介

本文档介绍了如何使用基于LSTM的深度学习模型进行人体跌倒检测。该模型利用3D人体姿态序列数据，通过时序特征学习来识别跌倒事件。该模型是作为FallDetection-Test-HoT项目的一部分开发的，与现有的3D人体姿态估计功能完美集成。

## 模型架构

该跌倒检测模型基于双向LSTM网络，并结合了注意力机制，具有以下特点：

- **输入**：3D人体姿态序列数据 `[batch_size, sequence_length, feature_dim]`
- **输出**：二分类结果（跌倒/非跌倒）
- **核心组件**：
  - 双向LSTM层：捕获时序特征
  - 注意力机制：聚焦关键帧
  - 多层分类器：进行最终预测

## 数据准备

### 数据格式

训练模型需要准备以下数据：

1. **3D姿态数据**：使用项目现有的2D到3D姿态预测模块生成，保存为NPZ格式
   - 路径格式：`{data_dir}/{video_id}/output_3D/output_keypoints_3d.npz`

2. **标签文件**：CSV格式，至少包含以下字段：
   - `video_id`：视频ID，对应3D姿态数据目录
   - `has_fall`：是否包含跌倒（1表示跌倒，0表示非跌倒）
   - `fall_start_frame`：跌倒开始帧（可选，仅对跌倒视频）
   - `fall_end_frame`：跌倒结束帧（可选，仅对跌倒视频）

标签文件示例：

```csv
video_id,has_fall,fall_start_frame,fall_end_frame
sample_video,1,45,75
sample_video2,0,,
```

### 数据集生成

模型使用`PoseSequenceDataset`类处理数据，该类会：

- 根据指定的序列长度和滑动步长切分姿态序列
- 对于跌倒视频，确保跌倒事件被完整捕获
- 对于非跌倒视频，使用滑动窗口生成多个样本
- 自动平衡正负样本比例，避免类别不平衡

## 模型训练

### 命令行参数

训练脚本支持以下命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `./demo/output/` | 3D姿态数据目录 |
| `--label_file` | str | (必需) | 标签文件路径(CSV格式) |
| `--normal_seq_length` | int | 30 | 非跌倒视频序列长度 |
| `--normal_stride` | int | 20 | 非跌倒视频滑动步长 |
| `--fall_seq_length` | int | 30 | 跌倒视频序列长度 |
| `--fall_stride` | int | 15 | 跌倒视频滑动步长 |
| `--overlap_threshold` | float | 0.3 | 跌倒判定的重叠比例阈值 |
| `--batch_size` | int | 16 | 批大小 |
| `--num_epochs` | int | 50 | 训练轮数 |
| `--learning_rate` | float | 0.001 | 学习率 |
| `--weight_decay` | float | 1e-5 | 权重衰减 |
| `--hidden_dim` | int | 256 | LSTM隐藏层维度 |
| `--num_layers` | int | 3 | LSTM层数 |
| `--dropout` | float | 0.3 | Dropout比例 |
| `--save_dir` | str | `./checkpoints/fall_detection_lstm/` | 模型保存目录 |
| `--gpu` | str | '0' | GPU ID |
| `--seed` | int | 42 | 随机种子 |
| `--test_only` | flag | False | 仅测试模式 |
| `--checkpoint` | str | None | 加载checkpoint路径 |

### 训练示例

基础训练命令：

```bash
python train_lstm.py --label_file fall_frame_data.csv
```

使用自定义参数：

```bash
python train_lstm.py --label_file fall_frame_data.csv --data_dir ./demo/output/ \
                    --normal_seq_length 30 --normal_stride 20 \
                    --fall_seq_length 30 --fall_stride 15 \
                    --overlap_threshold 0.3 \
                    --batch_size 8 --num_epochs 100 \
                    --learning_rate 0.0005 --hidden_dim 256 \
                    --num_layers 3 --dropout 0.3 \
                    --save_dir ./my_models/fall_detection/
```

### 数据集生成策略

数据集生成采用了两种不同的策略：

1. **非跌倒视频序列生成**：
   - 使用固定序列长度(`normal_seq_length`)
   - 使用较大的滑动步长(`normal_stride`)
   - 生成的所有序列样本均标记为非跌倒(0)

2. **跌倒视频序列生成**：
   - 使用固定序列长度(`fall_seq_length`)
   - 使用较小的滑动步长(`fall_stride`)
   - 根据重叠比例判断是否为跌倒序列：
     - 计算当前序列窗口与跌倒帧段的重叠长度
     - 计算重叠比例 = 重叠长度 / 序列长度
     - 只有当重叠比例 ≥ `overlap_threshold` 时才标记为跌倒样本(1)
     - 这样可以避免边缘效应带来的错误标记

### 训练过程

训练过程中会：

1. 自动将数据集划分为训练集（60%）、验证集（20%）和测试集（20%）
2. 每个epoch结束后显示训练和验证的损失与准确率
3. 使用ReduceLROnPlateau调整学习率，在验证损失不再下降时降低学习率
4. 保存最佳模型（基于验证损失）
5. 每10个epoch保存一个检查点
6. 绘制训练和验证的损失与准确率曲线

## 模型评估

### 评估方法

评估脚本会：

1. 计算测试集上的损失
2. 生成分类报告，包括精确率、召回率和F1分数
3. 创建混淆矩阵并以图像形式保存
4. 保存详细的评估结果到pickle文件中，便于后续分析

### 仅评估模式

使用`--test_only`参数可以仅进行评估而不训练：

```bash
python train_fall_detection.py --label_file path/to/labels.csv \
                               --checkpoint ./checkpoints/fall_detection/best_model.pth \
                               --test_only
```

## 输出文件

训练与评估过程会生成以下文件：

- **模型检查点**：
  - `best_model.pth`：最佳性能的模型
  - `model_epoch_N.pth`：每10个epoch的检查点

- **训练历史**：
  - `training_history.pkl`：包含训练和验证的损失与准确率
  - `training_history.png`：训练曲线图

- **评估结果**：
  - `evaluation_results.pkl`：详细的评估指标和预测结果
  - `confusion_matrix.png`：混淆矩阵图像

## 实际应用

训练好的模型可以集成到现有的姿态估计系统中，实现：

1. 实时视频中的跌倒检测
2. 离线视频分析
3. 监控系统预警

## 注意事项

- 确保有足够的跌倒和非跌倒样本用于训练，类别平衡对性能影响较大
- 序列长度应根据视频帧率和跌倒持续时间适当调整
- 如果GPU内存不足，可以减小批量大小或序列长度
- 对于不同场景（如室内/室外、不同角度）的泛化能力，建议使用多样化的训练数据

## 故障排除

- **Out of Memory错误**：减小批量大小、序列长度或模型隐藏层维度
- **过拟合**：增加dropout比例，减小模型复杂度，或增加数据增强
- **欠拟合**：增加训练轮数，提高模型复杂度，或降低学习率
- **找不到标签文件或数据目录**：检查文件路径是否正确，路径中是否包含特殊字符

## 进阶优化

- **数据增强**：添加随机旋转、缩放、噪声等增强方法
- **迁移学习**：使用其他数据集预训练模型
- **模型集成**：结合多个模型的预测结果提高准确率