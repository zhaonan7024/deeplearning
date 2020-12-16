import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
#from FlattenLayer import FlattenLayer
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt

#对多分类任务中的模型评估隐藏层层数和隐藏单元个数对实验结果的影响,改变神经元个数
#多分类数据集问题
train_dataset = torchvision.datasets.MNIST(root='~/Datasets/MNIST',train=True,download=True,transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='~/Datasets/MNIST',train=False,transform=transforms.ToTensor())
#批量读取数据
batch_size = 64
train_iter = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
test_iter = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)

#模型参数定义和初始化
#修改num_hiddens 隐藏层神经元的个数
num_inputs,num_outputs,num_hiddens = 784,10,1500

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)

#使用ReLU作为激活函数
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs),
)

for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)

#定义随机梯度下降函数
def SGD(params,lr):
    for param in params:
        param.data -=lr*param.grad

#计算模型在某个数据集上的准确率
def evaluate_accuracy(data_iter,net,loss):
    acc_sum,n = 0.0,0
    test_l_sum = 0.0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        l = loss(net(X),y).sum()
        test_l_sum += l.item()
        n +=y.shape[0]
    return acc_sum/n,test_l_sum/n

num_epochs = 10
lr = 0.1
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr)

#定义模型训练函数
def train(net,mnist_train,mnist_test,loss,num_epochs,batch_size,params=None,lr = None,optimizer = None):
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_1_sum,train_acc_sum,n = 0.0,0.0,0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                SGD(params,lr)
            else:
                optimizer.step()
            train_1_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
        test_acc,test_1 = evaluate_accuracy(test_iter,net,loss)
        train_loss.append(train_1_sum/n)
        test_loss.append(test_1)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f' %(epoch+1,train_1_sum/n,train_acc_sum/n,test_acc))
    return train_loss,test_loss

train_loss,test_loss = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters(),lr,optimizer)

#绘制loss曲线
x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth = 1.5)
plt.plot(x,test_loss,label = "test_loss",linewidth = 1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
