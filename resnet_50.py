import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import tqdm

# cuda加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# 加载数据集
ckplus = datasets.ImageFolder(root='./data/fer2013', transform=data_transforms)

# 划分数据集
# 定义训练集和测试集的比例
train_ratio = 0.8
test_ratio = 1 - train_ratio

# 计算划分的大小
train_size = int(train_ratio * len(ckplus))
test_size = len(ckplus) - train_size

# 划分数据集
torch.manual_seed(44)
train_data, test_data = random_split(ckplus, [train_size, test_size])

# 创建DataLoader加载数据
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 定义数据标签
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        start_time = time.time()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

            #scheduler.step()

            epoch_loss = running_loss / train_size
            epoch_acc = running_corrects.double() / train_size

        end_time = time.time()
        print(f"训练时间:{end_time - start_time}秒")

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            right_num = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                right_num = torch.eq(preds, labels).sum().item() + right_num
            test_acc = right_num / len(test_data)
            print('Test_Accuracy :', test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print('Best Test Accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# 引入训练好的resnet50
model = models.resnet50(pretrained=True)
for name, param in model.named_parameters():
    if 'conv' in name:
        param.requires_grad = False

# 修改线性层使之符合任务需求
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(emotion_labels))
)

# 将模型移入cuda中
model = model.to(device)

# 定义损失函数、优化器以及动态学习率调整
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

# 保存模型
torch.save(model.state_dict(), 'resnet50_emotion.pth')

# 在测试集上进行验证
# model.eval()
# with torch.no_grad():
#     right_num = 0
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         right_num = torch.eq(preds, labels).sum().item()+right_num
#     acc = right_num/len(test_data)
#     print('Accuracy :', acc)

# 展示验证结果
# for i in range(5):
#     input_img = inputs[i].cpu().numpy().transpose((1, 2, 0))
#     plt.imshow(input_img)
#     plt.title('Predicted: {}'.format(emotion_labels[preds[i]]))
#     plt.show()
