#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='dataset/data.yaml', epochs=250, batch=8)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')



