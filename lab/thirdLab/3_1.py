import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time
import numpy as np
transforms.ToTensor()

#残差网络实验

#读取数据模块
transform = transforms.Compose(
    [transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）=（32，32）或者保持长宽比并缩放最短的边到int大小
     transforms.CenterCrop(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 将数据进行归一化

#获取数据集
train_path = r"./data/cars_new/train_set"
test_path = r"./data/cars_new/test_set"

train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transform)


classes = ('bus','car','truck')
num_classes = 3
lr = 0.001
batch_size = 512
device = torch.device("cpu")

# 生成dataloader
trainloadr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

#残差网络的实现
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock, self).__init__()
        #两层卷积
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        #输出层
        self.shortcut=nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        # 图片先经过三层卷积
        out = self.left(x)
        #实现残差
        out +=self.shortcut(x)
        out = F.relu(out)
        return out


# 残差网络
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=3):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#训练函数
def train_epoch(net,data_loader,device):
    net= net.to(device)
    net.train() #指定当前为训练模式
    train_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0

    #遍历每个batch进行训练
    for batch_idx,(data,target) in enumerate(data_loader):
        #将图片放入指定的device
        data = data.to(device).float()
        target = target.to(device).long()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        prediction = torch.argmax(output,1)
        correct +=(prediction == target).sum().item()
        sample_num+=len(prediction)
    loss = total_loss/train_batch_num
    acc = correct/sample_num
    return loss,acc

#测试函数
def tes_epoch(net,data_loader,device):
    net.eval()
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    #指定不进行梯度变化
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(data_loader):
            data = data.to(device).float()
            target = target.to(device).long()
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().item()
            sample_num += len(prediction)
        loss = total_loss / test_batch_num
        acc = correct / sample_num
        return loss, acc

#使用交叉熵损失
net = ResNet(ResidualBlock)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=lr)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
#进行训练
epochs=20

startTime = time.time()
for epoch in range(epochs):
    #在训练集上训练
    train_loss,train_acc = train_epoch(net,data_loader=trainloadr,device=device)
    #训练集上验证
    test_loss, test_acc = tes_epoch(net, data_loader=testloader, device=device)
    #保存各个指标
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    print(f"epoch:{epoch}\t train_loss:{train_loss:.4f} \t"
          f"train_acc:{train_acc} \t"
          f"test_loss:{test_loss:.4f} \t test_acc:{test_acc}")

endTime = time.time()
print("训练时间",endTime - startTime)

def plt_loss():
    #绘制loss曲线
    x = np.linspace(0,len(train_loss_list),len(train_loss_list))
    plt.plot(x,train_loss_list,label="train_loss",linewidth = 1.5)
    plt.plot(x,test_loss_list,label = "test_loss",linewidth = 1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
def plt_acc():
    #绘制精确率曲线
    x = np.linspace(0,len(train_acc_list),len(train_acc_list))
    plt.plot(x,train_acc_list,label="train_acc",linewidth = 1.5)
    plt.plot(x,test_acc_list,label = "test_acc",linewidth = 1.5)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
plt_loss()
plt_acc()
