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


#批量读取数据
batch_size = 64

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
def evaluate_accuracy(X,y,net,loss):
    n = 0
    test_l_sum = 0.0
    n+=y.shape[0]
    l = loss(net(X), y).sum()
    test_l_sum += l.item()
    return test_l_sum/n

#定义模型训练函数
def train(net,data,loss,num_epochs,params=None,lr = None,optimizer = None):
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum = 0.0,0.0
        n = 0
        X_train = data[0]
        y_train = data[1]
        X_valid = data[2]
        y_valid = data[3]
        y_hat = net(X_train)
        l = loss(y_hat,y_train).sum()
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
        n+=y_train.shape[0]
        test_1 = evaluate_accuracy(X_valid, y_valid, net, loss)
        train_loss.append(train_l_sum / n)
        test_loss.append(test_1)
    return np.mean(train_loss), np.mean(test_loss)

num_epochs = 30
lr = 0.03
batch_size = 1
#train_loss,test_loss=train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


#获取K折交叉验证某一折的训练集和验证集
def get_kfold_data(k,i,X,y):
    fold_size = X.shape[0] //k
    val_start = i*fold_size
    if i!=k-1:
        val_end = (i+1)*fold_size
        X_valid,y_valid = X[val_start:val_end],y[val_start:val_end]
        X_train = torch.cat((X[0:val_start],X[val_end:]),dim = 0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:
        X_valid,y_valid = X[val_start:],y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
    return  X_train,y_train,X_valid,y_valid

def k_fold(k,X_train,y_train):
    train_loss_sum,valid_loss_sum = 0,0
    for i in range(k):
        print('第',i+1,'折验证结果')
        data = get_kfold_data(k,i,X_train,y_train)
        train_loss, val_loss= train(net,data,loss,num_epochs,params,lr)
        print('train_loss: %.7f,val_loss:%.7f'%(train_loss,val_loss))
        train_loss_sum += train_loss
        valid_loss_sum +=val_loss
    return

k = 10
k_fold(k,features,labels)


