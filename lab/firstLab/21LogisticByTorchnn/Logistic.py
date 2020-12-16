import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
from collections import OrderedDict
import LogisticNet
from matplotlib import pyplot as plt


plt.legend(['InitialData','ResultData'])
num_inputs = 2
num_examples=1000
true_w=[2,-3.4]  #真实权重
true_b=4.2  #偏差
features=torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels = labels.exp() / (labels.exp() + 1)
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

def initialData(labels):
    for i in range(0,labels.shape[0]):
        if(labels[i]>0.5):
            labels[i]=1
        else:
            labels[i]=0
initialData(labels)

#读取数据
lr = 0.03
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers=0,              # 多线程来读数据，注意在Windows下需要设置为0)
)
print(dataset)
#将初始的数据打印出来
plt.scatter(features[:,1].numpy(),labels.numpy(),label='InitialData')
#plt.show()

num_outputs=1
#net =LogisticNet.LogisticNet()


net = nn.Sequential(
    OrderedDict([
        ('linear', nn.Linear(num_inputs, 1)),
        ('sigmoid', nn.Sigmoid())
    ])
)

#参数初始化
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)   #也可以直接修改bias的data
loss = nn.MSELoss()
#小批量随机梯度下降算法
learningRate = 0.005
optimizer = optim.SGD(net.parameters(), lr=learningRate)  #梯度下降的学习率指定为0.03
num_epochs = 30

#准确率计算函数
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

#评价模型 𝐧𝐞𝐭在数据集 𝐝𝐚𝐭𝐚_𝐢𝐭𝐞𝐫 上的准确率

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

print(true_w, '\n', net[0].weight)
print(true_b, '\n', net[0].bias)


#准确率
def accuracy(y_hat, y):
    for i in range(0, y_hat.shape[0]):
        if (y_hat[i][0] > 0.5):
            y_hat[i][0] = 1
        else:
            y_hat[i][0] = 0
  #  return (y_hat.argmax(dim=1) == y).float().mean().item()
    return (y_hat == y).float().mean().item()


def evaluate_accuracy(data_iter, net,w,b):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X,w,b).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        return acc_sum / n

labels_true=net[0].weight.T[0]*features[:,0]+net[0].weight.T[1]*features[:,1]+net[0].bias
labels_true = labels_true.exp() / (labels_true.exp() + 1)
labels_true+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
initialData(labels_true)
plt.scatter(features[:,1].numpy(),labels_true.detach().numpy(),color = 'r',label='ResultData')
plt.legend()
plt.show()
