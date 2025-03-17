import torch
import os

# 加载模型
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# 图像路径
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "dog.jpg")

# 检测结果
results = model(filename)  # inference
results.show()
