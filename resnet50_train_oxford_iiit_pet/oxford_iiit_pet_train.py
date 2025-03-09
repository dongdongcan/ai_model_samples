# 导入必要库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小至 224x224（ResNet标准输入）
        transforms.ToTensor(),  # 转换为 Tensor，值范围归一化至 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化（使用 ImageNet 的均值和标准差）
    ]
)

# 加载数据集（Oxford-IIIT Pet Dataset）
train_dataset = datasets.OxfordIIITPet(
    root="./data", split="trainval", target_types="category", download=True, transform=transform
)
test_dataset = datasets.OxfordIIITPet(
    root="./data", split="test", target_types="category", download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 初始化 ResNet50 模型
model = torchvision.models.resnet50(weights=None)  # 从零开始训练
model.fc = nn.Linear(model.fc.in_features, len(train_dataset._labels))  # 修改全连接层为 37 类
model = model.to(device)  # 转移到 GPU/CPU

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适用于多分类）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每 5 epoch 学习率衰减 10%

# 训练函数
def train(epoch):
    model.train()  # 训练模式
    total, correct = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 计算准确率
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print(f"Epoch {epoch} | Train Acc: {acc:.2f}%")


# 测试函数
def test(epoch):
    model.eval()  # 评估模式
    total, correct = 0, 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print(f"Epoch {epoch} | Test Acc: {acc:.2f}%")
    return acc


# 主训练循环
best_acc = 0.0
for epoch in range(1, 21):  # 训练 20 个 epoch
    start_time = time.time()
    train(epoch)
    test_acc = test(epoch)
    scheduler.step()  # 更新学习率

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_pet_model.pth")

    print(f"Epoch {epoch} Time: {time.time() - start_time:.2f}s")

print(f"Best Test Accuracy: {best_acc:.2f}%")

"""
关键参数说明：
- batch_size=32: 每批次样本数，平衡内存和训练效率
- num_workers=2: 数据加载并行进程数，根据 CPU 核心数调整
- lr=0.001: Adam 初始学习率
- StepLR: 每 5 epoch 学习率降为原来的 10%
- epochs=20: 训练轮次，适中选择

建议扩展：
- 使用预训练权重（weights='DEFAULT'）加速收敛
- 添加数据增强（如随机翻转、旋转）提升泛化性
- 使用 TensorBoard 可视化训练过程
"""
