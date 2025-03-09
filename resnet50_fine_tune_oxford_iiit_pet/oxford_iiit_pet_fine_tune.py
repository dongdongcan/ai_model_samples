import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

# 设备配置：优先使用GPU加速计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：与ImageNet预训练一致，确保输入兼容
transform = transforms.Compose(
    [
        transforms.Resize(256),  # 缩放至256x256，便于后续裁剪
        transforms.CenterCrop(224),  # 裁剪为224x224，匹配ResNet输入
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ]
)

# 加载Oxford-IIIT Pet数据集：37类宠物分类任务
train_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform)
test_dataset = datasets.OxfordIIITPet(root="./data", split="test", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 加载预训练ResNet50：利用ImageNet权重提高初始性能
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 冻结卷积层：保留预训练特征提取能力，仅微调深层和分类层
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:  # 只允许layer4和全连接层训练
        param.requires_grad = False

# 修改全连接层：适配37类输出
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)

# 优化器：为不同层设置不同学习率，加速收敛并避免破坏预训练权重
optimizer = optim.Adam(
    [
        {"params": model.fc.parameters(), "lr": 1e-3},  # 全连接层用较高学习率
        {"params": model.layer4.parameters(), "lr": 1e-5},  # layer4用较低学习率
    ]
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率衰减

# 训练函数：计算损失并更新参数
def train(epoch):
    model.train()
    total, correct = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f"Train Epoch: {epoch} | Acc: {100. * correct / total:.2f}%")


# 测试函数：评估模型性能
def test(epoch):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print(f"Test Epoch: {epoch} | Acc: {acc:.2f}%")
    return acc


# 训练循环：微调15个epoch，保存最佳模型
best_acc = 0.0
for epoch in range(1, 16):
    train(epoch)
    test_acc = test(epoch)
    scheduler.step()
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "fine_tuned_model.pth")

print(f"Best Test Accuracy: {best_acc:.2f}%")
