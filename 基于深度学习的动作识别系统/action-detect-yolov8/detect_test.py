from ultralytics import YOLO

# 所需加载的模型目录
path = 'models/123.pt'
# 加载预训练模型
# conf	0.25	object confidence threshold for detection
# iou	0.7	intersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')

# 需要检测的图片地址
img_path = '2pai.jpg'

results = model(img_path)[0]
results.plot()