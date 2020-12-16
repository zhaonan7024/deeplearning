import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
transforms.ToTensor()

#实现Alexnet
#读取数据模块
transform = transforms.Compose(
    [transforms.Resize((227, 227)),  # 将图片缩放到指定大小（h,w）=（327，327）或者保持长宽比并缩放最短的边到int大小
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
batch_size = 128
device = torch.device("cpu")
# 生成dataloader
trainloadr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4,padding=0), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。类别数为3，而非论文中的1000
            nn.Linear(4096, 3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 6 * 6 * 256)
        output = self.fc(x)
        return output


#训练函数
def train_epoch(net,data_loader,device):
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

net = AlexNet().to(device)
#使用交叉熵损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=lr)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
#进行训练
epochs=10

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
print('endtime',endTime)
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
