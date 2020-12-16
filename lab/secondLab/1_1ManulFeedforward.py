import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

#功能：手动实现前馈神经网络解决线性回归任务

#数据集
n_train, n_test, num_inputs = 7000, 3000, 500
true_w, true_b = torch.ones(num_inputs, 1) * 0.0056, 0.028
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


#批量读取数据
batch_size = 64
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
dataset = torch.utils.data.TensorDataset(test_features, test_labels)
test_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

#模型参数定义和初始化
num_inputs,num_outputs,num_hiddens = 500,1,256
W1 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_inputs)),dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)
W2 = torch.tensor(np.random.normal(0,0.01,(num_outputs,num_hiddens)),dtype=torch.float)
b2 = torch.zeros(num_outputs,dtype=torch.float)
params = [W1,b1,W2,b2]
for param in params:
    param.requires_grad_(requires_grad=True)

#定义激活函数 ReLU
def relu(X):
    return torch.max(input = X,other=torch.tensor(0.0))

#定义交叉损失函数
#loss = torch.nn.CrossEntropyLoss()
loss = torch.nn.MSELoss()
#定义模型
def net(X):
    X = X.view((-1,num_inputs))
    H = relu(torch.matmul(X,W1.t())+b1)
    return torch.matmul(H,W2.t())+b2

#定义随机梯度下降函数
def SGD(params,lr):
    for param in params:
        param.data -=lr*param.grad

#计算模型在某个数据集上的准确率
def evaluate_accuracy(data_iter,net,loss):
    acc_sum,n = 0.0,0
    test_l_sum = 0.0
    for X,y in data_iter:
  #      print('net X',net(X)-y)
        acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
        l = loss(net(X), y).sum()
        test_l_sum += l.item()
    return acc_sum/n,test_l_sum/n

#定义模型训练函数
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr = None,optimizer = None):
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum = 0.0,0.0
        n = 0
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
            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=0)==y).sum().item()
            n+=y.shape[0]
        test_acc,test_1 = evaluate_accuracy(test_iter,net,loss)
        train_loss.append(train_l_sum / n)
        test_loss.append(test_1)
        print('epoch %d,loss %.4f,test loss %.4f' % (epoch + 1, train_l_sum / n, test_1))
    return train_loss, test_loss

num_epochs = 30
lr = 0.03
batch_size = 1
train_loss,test_loss=train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


#绘制loss曲线
x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth = 1.5)
plt.plot(x,test_loss,label = "test_loss",linewidth = 1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
