# Deep Learning Action Recognition

基于深度学习的动作识别系统，使用 YOLOv8 进行人体姿态估计，结合深度学习模型进行动作分类。

## 功能特性

- 基于 YOLOv8 的人体姿态估计
- 多动作类别识别
- GPU加速推理
- 实时视频流处理
- 可定制的训练流程

## 技术栈

- **目标检测**: YOLOv8 (Ultralytics)
- **深度学习**: PyTorch
- **姿态估计**: YOLOv8 Pose
- **数据处理**: NumPy, OpenCV

## 快速启动

### 训练

```bash
python train_action_model.py
```

### 识别

```bash
python action_recognizer.py
```
