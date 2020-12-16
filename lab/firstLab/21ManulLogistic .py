import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from torch import nn
plt.legend(['InitialData','ResultData'])
num_inputs = 2
num_examples=1000
true_w=[2,-3.4]  #真实权重
true_b=4.2  #偏差
features=torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels = labels.exp() / (labels.exp() + 1)
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
#labels数据初始化为0,1
def initialData(labels):
    for i in range(0,labels.shape[0]):
        if(labels[i]>0.5):
            labels[i]=1
        else:
            labels[i]=0

initialData(labels)
plt.scatter(features[:,1].numpy(),labels.numpy(),label='InitialData')


def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)


w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def net(X,w,b):
    return ((torch.mm(X, w) + b).exp()/((torch.mm(X, w) + b).exp()+1))

def linreg(X, w, b):
    return (torch.mm(X, w) + b)
def logistic(Z):
    result = Z.exp()/(Z.exp()+1)
    return (result)


#梯度下降
def gradDescent(params, x , y, yt, learning_rate):
    params[0].data = params[0].data + learning_rate * (torch.matmul((y-yt.t()),x) ).t() / yt.shape[0]   # 更新参数w
    params[1].data = params[1].data + learning_rate * (y-yt).sum(axis=0)[0]   # 更新参数b

num_epochs = 10
batch_size = 10
learning_rate = 0.03
loss = nn.BCELoss()
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        result1 = linreg(X, w, b)
        y_hat=logistic(result1)
        l = loss(y_hat, y).sum()     # l是梯度下降的损失
    #    l = ((-(y * np.log(result2) + (1 - X) * np.log(1 - result2))).sum(axis=0)[0] / result2.shape[0])
        l.backward()     # 小批量的损失对模型参数求梯度
        gradDescent([w, b], X, y, y_hat, learning_rate)
        w.grad.data.zero_()    # 梯度清零
        b.grad.data.zero_()
    train_l = loss(logistic(linreg(features, w, b)), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)


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


#测试
num_Test=100
featuresTest=torch.tensor(np.random.normal(0,1,(num_Test,num_inputs)),dtype=torch.float)
labelsTest=true_w[0]*featuresTest[:,0]+true_w[1]*featuresTest[:,1]+true_b
labelsTest = labelsTest.exp() / (labelsTest.exp() + 1)
labelsTest+=torch.tensor(np.random.normal(0,0.01,size=labelsTest.size()),dtype=torch.float)

#labels数据初始化为0,1
for i in range(0,labelsTest.shape[0]):
    if(labelsTest[i]>0.5):
        labelsTest[i]=1
    else:
        labelsTest[i]=0

result = logistic(linreg(featuresTest, w, b))
print(accuracy(result, labelsTest))


#做出分界函数
labels_true=w[0]*features[:,0]+w[1]*features[:,1]+b
labels_true = labels_true.exp() / (labels_true.exp() + 1)
labels_true+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
initialData(labels_true)
plt.scatter(features[:,1].numpy(),labels_true.detach().numpy(),color = 'r',label='ResultData')
plt.legend()
plt.show()

