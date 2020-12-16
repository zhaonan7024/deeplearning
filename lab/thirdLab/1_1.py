import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import random
transforms.ToTensor()

#手写二维卷积的实现，并在至少一个数据集上进行实验，从训练时间、预测精度、Loss变化等角度分析实验结果（最好使用图表展示）

#读取数据模块
# 定义一个transform操作，用户将torch中的数据转换为可以输入到我们模型的形式
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
batch_size = 64
lr = 0.001
device = torch.device("cpu")

# 生成dataloader
trainloadr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


#卷积计算
def corr2d(X,K):
    batch_size, H,W = X.shape
    k_h,k_w = K.shape
    Y = torch.zeros((batch_size,H-k_h+1,W-k_w+1)).to(device)
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            Y[:,i,j] = (X[:,i:i+k_h,j:j+k_w]*K).sum()
    return Y

#实现多输入通道
def corr2d_multi_in(X,K):
    res = corr2d(X[:,0,:,:],K[0,:,:])
    for i in range(1,X.shape[1]):
        res += corr2d(X[:,i,:,:],K[i,:,:])
    return res

#实现多输出通道
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],dim = 1)


#自定义卷积层
class MyConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(MyConv2D,self).__init__()
        #初始化卷积层参数：卷积核、偏差
        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        self.weight = nn.Parameter(torch.randn((out_channels,in_channels)+kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels,1,1))
    def forward(self,x):
        return corr2d_multi_in_out(x,self.weight)+self.bias

class MyConvModule(nn.Module):
    def __init__(self):
        super(MyConvModule,self).__init__()
        #定义三层卷积层
        self.conv = nn.Sequential(
            MyConv2D(in_channels=3,out_channels=32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
      #      MyConv2D(in_channels=32, out_channels=64, kernel_size=3),
       #     nn.BatchNorm2d(64),
       #     nn.ReLU(inplace=True)
        )
        #输出层，将通道变为分类数量
        self.fc = nn.Linear(32,num_classes)
    #    # 池化操作
        self.pool = nn.AvgPool2d(30, stride=1)
    def forward(self,x):
        #图片先经过三层卷积
        out = self.conv(x)
        out = self.pool(out)
    #    out = F.avg_pool2d(out,30)
        out = out.squeeze()
        out = self.fc(out)
        return out

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

net=MyConvModule().to(device)
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

