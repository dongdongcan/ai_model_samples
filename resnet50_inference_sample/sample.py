# 这段代码使用了 PyTorch 和预训练的 ResNet50 模型来对图像进行分类。

# 它首先加载了预训练的 ResNet50 模型，然后对指定目录下的所有图片进行预处理，并使用模型进行预测。
# 最后，它输出每张图片预测的前五个最可能的类别

# 导入所需的库
import torch  # PyTorch
import heapq  # 用于获取最大的N个元素
from PIL import Image  # 用于处理图像
from torchvision import transforms  # 用于图像预处理
import os  # 用于获取当前脚本所在的目录
import sys  # 用于添加当前目录到 PYTHONPATH

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "picture", "dog.jpg")
class_file = os.path.join(current_dir, "imagenet_classes.txt")

# 将当前目录添加到 PYTHONPATH
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 加载预训练的ResNet50模型
resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
resnet50.eval()  # 将模型设置为评估模式

# 定义图片目录
# filename = os.path.abspath("./picture/dog.jpg")
# 打开图像文件
input_image = Image.open(filename)
# 定义预处理步骤
preprocess = transforms.Compose(
    [
        transforms.Resize(224),  # 将图像大小调整为224x224
        # transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ]
)
# 对图像进行预处理
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# 使用模型对图像进行预测
output = resnet50(input_batch)

# 获取前五个最高概率的预测结果
res = list(output[0].detach().numpy())
index = heapq.nlargest(5, range(len(res)), res.__getitem__)

print("\npredict picture: " + filename)
# 读取类别名称
with open(class_file, "r") as f:
    categories = [s.strip() for s in f.readlines()]
    for i in range(5):
        print("         top " + str(i + 1) + ": " + categories[index[i]])
